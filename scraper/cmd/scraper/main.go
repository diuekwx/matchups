// Command scraper crawls op.gg's champion counters pages for LoL matchup
// data and writes the results as JSON chunks (see project brief for the
// eventual Postgres schema these map onto).
package main

import (
	"context"
	"flag"
	"log"
	"time"

	"matchup-scraper/internal/champion"
	"matchup-scraper/internal/opgg"
	"matchup-scraper/internal/pipeline"
	"matchup-scraper/internal/store"
)

func main() {
	championsFile := flag.String("champions", "data/champions.json", "path to the champion seed roster")
	slugsFile := flag.String("slugs", "data/opgg_champion_slugs.json", "path to the verified op.gg champion slug list")
	outFile := flag.String("out", "output/chunks.json", "path to write scraped chunks as JSON")
	reportFile := flag.String("report", "", "optional path to write a crawl report as JSON")
	requestsPerSecond := flag.Float64("rate", 2.0, "max requests/sec against op.gg (shared across all workers)")
	maxRetries := flag.Int("retries", 3, "max retries per page on failure, with exponential backoff")
	indexWorkers := flag.Int("index-workers", 3, "concurrent workers fetching counters index pages")
	detailWorkers := flag.Int("detail-workers", 3, "concurrent workers fetching matchup detail pages")
	maxOpponents := flag.Int("max-opponents", 15, "max opponents to deep-fetch per champion/role, by sample size")
	timeout := flag.Duration("timeout", 10*time.Minute, "overall crawl timeout")
	flag.Parse()

	seeds, err := champion.LoadSeeds(*championsFile)
	if err != nil {
		log.Fatalf("loading champion seeds: %v", err)
	}
	slugs, err := champion.LoadSlugRegistry(*slugsFile)
	if err != nil {
		log.Fatalf("loading slug registry: %v", err)
	}

	client := opgg.NewClient(*requestsPerSecond, *maxRetries)

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	log.Printf("crawling %d champions...", len(seeds))
	result := pipeline.Run(ctx, client, slugs, seeds, pipeline.Config{
		IndexWorkers:        *indexWorkers,
		DetailWorkers:       *detailWorkers,
		MaxOpponentsPerRole: *maxOpponents,
		RequestsPerSecond:   *requestsPerSecond,
		MaxRetries:          *maxRetries,
		Timeout:             timeout.String(),
	})
	log.Printf("scraped %d matchup chunks (%.1f%% detail completion)", len(result.Chunks), result.Report.CompletionRatePercent)

	if err := store.WriteChunksJSON(*outFile, result.Chunks); err != nil {
		log.Fatalf("writing output: %v", err)
	}
	log.Printf("wrote %s", *outFile)

	if *reportFile != "" {
		if err := store.WriteJSON(*reportFile, result.Report); err != nil {
			log.Fatalf("writing crawl report: %v", err)
		}
		log.Printf("wrote %s", *reportFile)
	}
}
