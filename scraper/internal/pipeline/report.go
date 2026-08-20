package pipeline

import (
	"sync"
	"time"
)

// CrawlReport summarizes the completeness and data quality of one crawl.
type CrawlReport struct {
	StartedAt             time.Time      `json:"started_at"`
	FinishedAt            time.Time      `json:"finished_at"`
	DurationMilliseconds  int64          `json:"duration_ms"`
	Config                Config         `json:"config"`
	Counts                CrawlCounts    `json:"counts"`
	CompletionRatePercent float64        `json:"completion_rate_percent"`
	Failures              []CrawlFailure `json:"failures"`
}

// CrawlCounts holds stage-level outcomes and chunk quality signals.
type CrawlCounts struct {
	IndexAttempted      int `json:"index_attempted"`
	IndexSucceeded      int `json:"index_succeeded"`
	IndexFailed         int `json:"index_failed"`
	IndexNoData         int `json:"index_no_data"`
	OpponentsDiscovered int `json:"opponents_discovered"`
	UnresolvedSlugs     int `json:"unresolved_slugs"`
	DetailAttempted     int `json:"detail_attempted"`
	DetailSucceeded     int `json:"detail_succeeded"`
	DetailFailed        int `json:"detail_failed"`
	DetailNoData        int `json:"detail_no_data"`
	EmptyTips           int `json:"empty_tips"`
	ZeroStatChunks      int `json:"zero_stat_chunks"`
	ChunksProduced      int `json:"chunks_produced"`
}

// CrawlFailure preserves enough context to reproduce a failed request.
type CrawlFailure struct {
	Stage     string `json:"stage"`
	Champion  string `json:"champion"`
	Opponent  string `json:"opponent,omitempty"`
	Role      string `json:"role"`
	SourceURL string `json:"source_url"`
	Error     string `json:"error"`
}

type reporter struct {
	mu     sync.Mutex
	report CrawlReport
}

func newReporter(cfg Config) *reporter {
	return &reporter{report: CrawlReport{
		StartedAt: time.Now().UTC(),
		Config:    cfg,
		Failures:  make([]CrawlFailure, 0),
	}}
}

func (r *reporter) update(fn func(*CrawlReport)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	fn(&r.report)
}

func (r *reporter) finish() CrawlReport {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.report.FinishedAt = time.Now().UTC()
	r.report.DurationMilliseconds = r.report.FinishedAt.Sub(r.report.StartedAt).Milliseconds()
	if r.report.Counts.DetailAttempted > 0 {
		r.report.CompletionRatePercent = 100 * float64(r.report.Counts.DetailSucceeded) / float64(r.report.Counts.DetailAttempted)
	}
	return r.report
}
