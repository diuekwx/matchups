package pipeline

import (
	"testing"
)

func TestReporterFinishCalculatesCompletionRate(t *testing.T) {
	r := newReporter(Config{IndexWorkers: 1, DetailWorkers: 2, Timeout: "2m0s"})
	r.update(func(report *CrawlReport) {
		report.Counts.DetailAttempted = 4
		report.Counts.DetailSucceeded = 3
	})

	report := r.finish()
	if report.CompletionRatePercent != 75 {
		t.Fatalf("CompletionRatePercent = %v, want 75", report.CompletionRatePercent)
	}
	if report.FinishedAt.Before(report.StartedAt) {
		t.Fatalf("FinishedAt %v is before StartedAt %v", report.FinishedAt, report.StartedAt)
	}
	if report.DurationMilliseconds < 0 {
		t.Fatalf("DurationMilliseconds = %d, want non-negative", report.DurationMilliseconds)
	}
	if report.Failures == nil {
		t.Fatal("Failures is nil, want an empty JSON array")
	}
}
