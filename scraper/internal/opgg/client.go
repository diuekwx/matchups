// Package opgg fetches and parses League of Legends matchup pages from
// op.gg. Verified before writing this: op.gg's robots.txt allows "*"
// user-agents with no AI-crawler carve-out, and the counters pages are
// server-rendered (the tip text and stats are present in the raw HTML,
// no JS execution required), so a plain HTTP client is sufficient.
package opgg

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/time/rate"
)

const userAgent = "matchup-rag-scraper/0.1 (+https://github.com/jerisonw/matchups; educational/portfolio project)"

// Client fetches op.gg pages under a shared rate limit, with retry and
// exponential backoff on transient failures.
type Client struct {
	http    *http.Client
	limiter *rate.Limiter
	retries int
}

// NewClient returns a Client limited to ratePerSecond requests/sec (shared
// across all callers) with up to maxRetries attempts per page on failure.
func NewClient(ratePerSecond float64, maxRetries int) *Client {
	return &Client{
		http:    &http.Client{Timeout: 15 * time.Second},
		limiter: rate.NewLimiter(rate.Limit(ratePerSecond), 1),
		retries: maxRetries,
	}
}

// Get fetches url, waiting on the rate limiter and retrying with
// exponential backoff on non-2xx responses or network errors.
func (c *Client) Get(ctx context.Context, url string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt <= c.retries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		if err := c.limiter.Wait(ctx); err != nil {
			return nil, err
		}

		body, err := c.doGet(ctx, url)
		if err == nil {
			return body, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("GET %s failed after %d attempts: %w", url, c.retries+1, lastErr)
}

func (c *Client) doGet(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "text/html")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}
