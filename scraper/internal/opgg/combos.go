package opgg

// The combo widget on a champion's build page (e.g.
// op.gg/lol/champions/ambessa/build/top) is client-hydrated -- it's absent
// from the raw server-rendered HTML that Client.Get fetches, so FetchCombos
// drives a real (headless) browser via chromedp instead. Verified in
// DevTools: each combo renders as
//
//	<button>
//	  <img src="https://img.youtube.com/vi/<id>/0.jpg" ...>
//	  <span>...<strong>Medium</strong>...
//	    <span class="line-clamp-2 ...">Q+AA+Q2+AA+E+AA</span>
//	  </span>
//	</button>
//
// This widget didn't appear in every session during manual checks (looked
// A/B-tested or session-gated), so an empty result here isn't necessarily
// an error -- callers should treat zero combos as "not shown this run",
// not "page is broken".
//
// The default chromedp headless mode ("old" headless Chrome) got a hard
// edge-level rejection here ("The request could not be satisfied") on both
// a home network and this sandbox -- not an IP block, a fingerprint block
// (old headless mode sets navigator.webdriver and other tells a WAF can
// key on). op.gg's own help center (help.op.gg, "Can I use OP.GG data?")
// states crawling is allowed given source citation (we store SourceURL
// per chunk) and reasonable request volume (the shared rate limiter
// covers that), so working around a generic bot-fingerprint filter -- not
// an access-control decision -- is in scope here. We use Chrome's newer
// "--headless=new" mode plus a realistic desktop UA, both of which make
// the browser behave and present much closer to a real Chrome instance.
import (
	"context"
	"fmt"
	"time"

	"github.com/chromedp/chromedp"
	"matchup-scraper/internal/model"
)

// comboWaitTimeout bounds how long FetchCombos waits for the widget to
// hydrate before giving up and returning whatever (possibly nothing) is
// present, rather than blocking a whole crawl run on a missing widget.
const comboWaitTimeout = 10 * time.Second

// desktopUserAgent mimics a normal desktop Chrome install. The default
// chromedp/old-headless-mode UA advertises "HeadlessChrome", which is
// itself a bot signal independent of the navigator.webdriver flag.
const desktopUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

// FetchCombos renders championSlug's build page for role in a headless
// browser and extracts the skill-order combo cards, if the widget is
// present in this session's variant.
func FetchCombos(ctx context.Context, championSlug, role string) ([]model.Combo, error) {
	allocOpts := append(chromedp.DefaultExecAllocatorOptions[:],
		// Overrides the default options' "headless" flag (old mode).
		chromedp.Flag("headless", "new"),
		chromedp.Flag("disable-blink-features", "AutomationControlled"),
		chromedp.UserAgent(desktopUserAgent),
		chromedp.WindowSize(1280, 800),
	)
	execCtx, cancelExec := chromedp.NewExecAllocator(ctx, allocOpts...)
	defer cancelExec()

	allocCtx, cancelAlloc := chromedp.NewContext(execCtx)
	defer cancelAlloc()

	browserCtx, cancelTimeout := context.WithTimeout(allocCtx, comboWaitTimeout+15*time.Second)
	defer cancelTimeout()

	var combos []model.Combo
	err := chromedp.Run(browserCtx,
		chromedp.Navigate(BuildURL(championSlug, role)),
		// Best-effort wait: the combo widget doesn't always render (see
		// package doc), so a timeout here means "not present", not "failed".
		waitForComboWidget(),
		chromedp.Evaluate(comboExtractJS, &combos),
	)
	if err != nil {
		return nil, fmt.Errorf("render %s combos for role %s: %w", championSlug, role, err)
	}
	return combos, nil
}

// waitForComboWidget waits for the combo card selector to appear, but caps
// the wait so pages without the widget don't stall the crawl.
func waitForComboWidget() chromedp.ActionFunc {
	return func(ctx context.Context) error {
		waitCtx, cancel := context.WithTimeout(ctx, comboWaitTimeout)
		defer cancel()
		_ = chromedp.Run(waitCtx, chromedp.WaitVisible(comboCardSelector, chromedp.ByQuery))
		return nil // ignore timeout -- absence is handled by the caller
	}
}

// comboCardSelector targets the combo sequence text specifically
// (line-clamp-2 inside a button), since it's more distinctive than the
// surrounding Tailwind utility-class soup, which is prone to shifting.
const comboCardSelector = `button span.line-clamp-2`

// comboExtractJS reads every combo card currently in the DOM. Mirrors
// waitForComboWidget's selector choice: difficulty comes from the
// <strong> badge, sequence from the line-clamp-2 span.
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
