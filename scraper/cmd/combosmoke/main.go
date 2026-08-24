// Throwaway smoke test for opgg.FetchCombos — not part of the pipeline.
// Run with: go run ./cmd/combosmoke
// Delete this directory (and the sibling .combo-profile dir it creates)
// once you're done debugging.
//
// Tests whether the combo widget is gated behind a sticky anonymous
// experiment-bucket cookie: reuses the same on-disk Chrome profile across
// several reloads (instead of a fresh throwaway profile each time, which
// is what FetchCombos and every earlier smoke test used) so any cookie
// op.gg sets on first visit survives to the next reload.
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/chromedp/chromedp"
)

const desktopUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

// profileDir is a persistent Chrome user-data dir living next to this
// throwaway command so cookies survive between process runs, not just
// between reloads within one run.
const profileDir = `.combo-profile`

func main() {
	for attempt := 1; attempt <= 5; attempt++ {
		combos, err := tryOnce(attempt)
		if err != nil {
			fmt.Printf("attempt %d: ERROR: %v\n", attempt, err)
			continue
		}
		fmt.Printf("attempt %d: combo cards = %d\n", attempt, combos)
		if combos > 0 {
			fmt.Println("got it -- stopping early")
			return
		}
		time.Sleep(2 * time.Second)
	}
}

func tryOnce(attempt int) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	allocOpts := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.Flag("headless", "new"),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.UserAgent(desktopUserAgent),
		chromedp.WindowSize(1280, 800),
		chromedp.UserDataDir(profileDir),
	)
	execCtx, cancelExec := chromedp.NewExecAllocator(ctx, allocOpts...)
	defer cancelExec()

	browserCtx, cancelBrowser := chromedp.NewContext(execCtx)
	defer cancelBrowser()

	var comboCount int
	err := chromedp.Run(browserCtx,
		chromedp.Navigate("https://op.gg/lol/champions/ambessa/build/top"),
		chromedp.Sleep(4*time.Second),
		chromedp.Evaluate(`document.querySelectorAll('button span.line-clamp-2').length`, &comboCount),
	)
	if err != nil {
		return 0, err
	}
	return comboCount, nil
}
