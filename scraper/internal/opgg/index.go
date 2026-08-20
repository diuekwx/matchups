package opgg

import (
	"bytes"
	"context"
	"errors"
	"strconv"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

// ErrNoData means the page loaded but doesn't have a matchup list for this
// champion/role (op.gg redirects off-role pages to the champion's build
// page instead of 404ing, so we detect it by content, not status code).
var ErrNoData = errors.New("no matchup data on page (champion likely not played in this role)")

// OpponentSummary is one row from a champion's counters list: an opponent
// plus their aggregate win rate and sample size against this champion.
type OpponentSummary struct {
	Name    string
	WinRate string
	Games   int
}

// FetchIndex fetches and parses the counters list for championSlug in role.
func (c *Client) FetchIndex(ctx context.Context, championSlug, role string) ([]OpponentSummary, error) {
	body, err := c.Get(ctx, IndexURL(championSlug, role))
	if err != nil {
		return nil, err
	}
	return ParseIndex(body)
}

// ParseIndex extracts the opponent list from a counters page's HTML.
func ParseIndex(html []byte) ([]OpponentSummary, error) {
	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(html))
	if err != nil {
		return nil, err
	}

	// The opponent search/filter box only renders on pages that actually
	// have matchup data for this champion+role; its absence means op.gg
	// redirected us to the champion's build page instead.
	if doc.Find(`label[for="championSearchAndFilter"]`).Length() == 0 {
		return nil, ErrNoData
	}

	var opponents []OpponentSummary
	doc.Find("li").Each(func(_ int, li *goquery.Selection) {
		img := li.Find("img[alt]").First()
		name, hasName := img.Attr("alt")
		winRateText := strings.TrimSpace(li.Find("strong").First().Text())
		if !hasName || name == "" || winRateText == "" {
			return
		}

		gamesText := strings.TrimSpace(li.Find("span.text-gray-600").Last().Text())
		games, _ := strconv.Atoi(strings.ReplaceAll(gamesText, ",", ""))

		opponents = append(opponents, OpponentSummary{
			Name:    name,
			WinRate: strings.ReplaceAll(winRateText, " ", ""),
			Games:   games,
		})
	})
	return opponents, nil
}
