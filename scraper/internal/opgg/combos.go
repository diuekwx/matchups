package opgg

import (
	"context"
	"fmt"
	"time"

	"matchup-scraper/internal/model"

	"github.com/chromedp/chromedp"
)

const (
	comboScrollSteps = 12
	comboScrollPause = 600 * time.Millisecond
	comboFetchBudget = 20 * time.Second
)

const desktopUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

func FetchCombos(ctx context.Context, championSlug, role string) ([]model.Combo, error) {
	allocOpts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", "new"),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.UserAgent(desktopUserAgent),
		chromedp.WindowSize(1280, 800),
	)
	execCtx, cancelExec := chromedp.NewExecAllocator(ctx, allocOpts...)
	defer cancelExec()

	allocCtx, cancelAlloc := chromedp.NewContext(execCtx)
	defer cancelAlloc()

	browserCtx, cancelTimeout := context.WithTimeout(allocCtx, comboFetchBudget)
	defer cancelTimeout()

	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(BuildURL(championSlug, role)),
		chromedp.Sleep(2*time.Second),
	); err != nil {
		return nil, fmt.Errorf("load %s build page for role %s: %w", championSlug, role, err)
	}

	var combos []model.Combo
	for step := 0; step < comboScrollSteps; step++ {
		if err := chromedp.Run(browserCtx,
			chromedp.Evaluate(`window.scrollBy(0, window.innerHeight * 0.8)`, nil),
			chromedp.Sleep(comboScrollPause),
			chromedp.Evaluate(comboExtractJS, &combos),
		); err != nil {
			return nil, fmt.Errorf("scroll/extract %s combos for role %s: %w", championSlug, role, err)
		}
		if len(combos) > 0 {
			break
		}
	}
	return combos, nil
}

const comboCardSelector = `button span.line-clamp-2`

const comboExtractJS = `
Array.from(document.querySelectorAll(` + "`" + comboCardSelector + "`" + `)).map(seqEl => {
  const card = seqEl.closest('button');
  const strongEl = card ? card.querySelector('strong') : null;
  return {
    difficulty: strongEl ? strongEl.textContent.trim() : '',
    sequence: seqEl.textContent.trim(),
  };
}).filter(c => c.sequence !== '')
`
