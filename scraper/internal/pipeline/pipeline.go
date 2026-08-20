// Package pipeline runs the two-stage scrape: for each seed champion/role,
// fetch its counters index to discover real opponents (rather than
// assuming every pair in the seed list is a meaningful matchup), then
// fetch each opponent's detail panel for the tip text and stats.
package pipeline

import (
	"context"
	"errors"
	"log"
	"sort"
	"sync"
	"time"

	"matchup-scraper/internal/champion"
	"matchup-scraper/internal/model"
	"matchup-scraper/internal/opgg"
)

// Config controls pipeline concurrency and crawl breadth.
type Config struct {
	IndexWorkers        int     `json:"index_workers"`          // concurrent workers fetching counters index pages
	DetailWorkers       int     `json:"detail_workers"`         // concurrent workers fetching matchup detail pages
	MaxOpponentsPerRole int     `json:"max_opponents_per_role"` // cap on opponents deep-fetched per champion/role, by sample size
	RequestsPerSecond   float64 `json:"requests_per_second"`
	MaxRetries          int     `json:"max_retries"`
	Timeout             string  `json:"timeout"`
}

// Result contains both successful chunks and the crawl's completeness report.
type Result struct {
	Chunks []model.Chunk
	Report CrawlReport
}

type indexJob struct {
	Champion champion.Seed
	Role     champion.Role
}

type detailJob struct {
	ChampionSlug string
	ChampionName string
	OpponentSlug string
	OpponentName string
	Role         champion.Role
}

// Run crawls every (champion, role) pair in seeds and returns one Chunk
// per resolved matchup.
func Run(ctx context.Context, client *opgg.Client, slugs *champion.SlugRegistry, seeds []champion.Seed, cfg Config) Result {
	indexJobs := make(chan indexJob)
	detailJobs := make(chan detailJob)
	chunks := make(chan model.Chunk)
	report := newReporter(cfg)

	var indexWG, detailWG sync.WaitGroup

	// Stage 1: champion/role -> discover real opponents -> emit detail jobs.
	for i := 0; i < cfg.IndexWorkers; i++ {
		indexWG.Add(1)
		go func() {
			defer indexWG.Done()
			for job := range indexJobs {
				runIndexJob(ctx, client, slugs, job, cfg.MaxOpponentsPerRole, detailJobs, report)
			}
		}()
	}

	// Stage 2: detail job -> fetch tip + stats -> emit a chunk.
	for i := 0; i < cfg.DetailWorkers; i++ {
		detailWG.Add(1)
		go func() {
			defer detailWG.Done()
			for job := range detailJobs {
				runDetailJob(ctx, client, job, chunks, report)
			}
		}()
	}

	go func() {
		for _, s := range seeds {
			for _, r := range s.Roles {
				select {
				case indexJobs <- indexJob{Champion: s, Role: r}:
				case <-ctx.Done():
					close(indexJobs)
					indexWG.Wait()
					close(detailJobs)
					detailWG.Wait()
					close(chunks)
					return
				}
			}
		}
		close(indexJobs)
		indexWG.Wait()
		close(detailJobs)
		detailWG.Wait()
		close(chunks)
	}()

	var results []model.Chunk
	for c := range chunks {
		results = append(results, c)
	}
	return Result{Chunks: results, Report: report.finish()}
}

func runIndexJob(ctx context.Context, client *opgg.Client, slugs *champion.SlugRegistry, job indexJob, maxOpponents int, detailJobs chan<- detailJob, report *reporter) {
	report.update(func(r *CrawlReport) { r.Counts.IndexAttempted++ })
	opponents, err := client.FetchIndex(ctx, job.Champion.Slug, string(job.Role))
	if err != nil {
		report.update(func(r *CrawlReport) {
			if errors.Is(err, opgg.ErrNoData) {
				r.Counts.IndexNoData++
			} else {
				r.Counts.IndexFailed++
			}
			r.Failures = append(r.Failures, CrawlFailure{Stage: "index", Champion: job.Champion.Name, Role: string(job.Role), SourceURL: opgg.IndexURL(job.Champion.Slug, string(job.Role)), Error: err.Error()})
		})
		log.Printf("index %s/%s: %v", job.Champion.Slug, job.Role, err)
		return
	}
	report.update(func(r *CrawlReport) { r.Counts.IndexSucceeded++ })

	sort.Slice(opponents, func(i, j int) bool { return opponents[i].Games > opponents[j].Games })
	if len(opponents) > maxOpponents {
		opponents = opponents[:maxOpponents]
	}
	report.update(func(r *CrawlReport) { r.Counts.OpponentsDiscovered += len(opponents) })

	for _, o := range opponents {
		slug, ok := slugs.Resolve(o.Name)
		if !ok {
			report.update(func(r *CrawlReport) {
				r.Counts.UnresolvedSlugs++
				r.Failures = append(r.Failures, CrawlFailure{Stage: "slug_resolution", Champion: job.Champion.Name, Opponent: o.Name, Role: string(job.Role), SourceURL: opgg.IndexURL(job.Champion.Slug, string(job.Role)), Error: "unresolved opponent slug"})
			})
			log.Printf("skip opponent %q vs %s/%s: unresolved slug", o.Name, job.Champion.Slug, job.Role)
			continue
		}
		select {
		case detailJobs <- detailJob{
			ChampionSlug: job.Champion.Slug,
			ChampionName: job.Champion.Name,
			OpponentSlug: slug,
			OpponentName: o.Name,
			Role:         job.Role,
		}:
		case <-ctx.Done():
			return
		}
	}
}

func runDetailJob(ctx context.Context, client *opgg.Client, job detailJob, chunks chan<- model.Chunk, report *reporter) {
	report.update(func(r *CrawlReport) { r.Counts.DetailAttempted++ })
	detail, err := client.FetchDetail(ctx, job.ChampionSlug, string(job.Role), job.OpponentSlug)
	if err != nil {
		report.update(func(r *CrawlReport) {
			if errors.Is(err, opgg.ErrNoData) {
				r.Counts.DetailNoData++
			} else {
				r.Counts.DetailFailed++
			}
			r.Failures = append(r.Failures, CrawlFailure{Stage: "detail", Champion: job.ChampionName, Opponent: job.OpponentName, Role: string(job.Role), SourceURL: opgg.DetailURL(job.ChampionSlug, string(job.Role), job.OpponentSlug), Error: err.Error()})
		})
		log.Printf("detail %s vs %s/%s: %v", job.ChampionSlug, job.OpponentSlug, job.Role, err)
		return
	}

	chunk := model.Chunk{
		Champion:  job.ChampionName,
		Opponent:  job.OpponentName,
		Role:      string(job.Role),
		SourceURL: opgg.DetailURL(job.ChampionSlug, string(job.Role), job.OpponentSlug),
		Tip:       detail.Tip,
		Stats:     detail.Stats,
		CreatedAt: time.Now().UTC(),
	}
	chunk.ChunkText = model.BuildChunkText(chunk.Champion, chunk.Opponent, chunk.Role, chunk.Tip, chunk.Stats)
	select {
	case chunks <- chunk:
		report.update(func(r *CrawlReport) {
			r.Counts.DetailSucceeded++
			r.Counts.ChunksProduced++
			if chunk.Tip == "" {
				r.Counts.EmptyTips++
			}
			if len(chunk.Stats) == 0 {
				r.Counts.ZeroStatChunks++
			}
		})
	case <-ctx.Done():
	}
}
