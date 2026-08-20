package opgg

import (
	"bytes"
	"context"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"matchup-scraper/internal/model"
)

// Detail is one champion-vs-opponent matchup panel: the write-up tip plus
// the head-to-head stat rows (win rate, KDA, kill participation, etc).
type Detail struct {
	Tip   string
	Stats []model.Stat
}

// FetchDetail fetches and parses the matchup panel for championSlug vs
// opponentSlug in role.
func (c *Client) FetchDetail(ctx context.Context, championSlug, role, opponentSlug string) (Detail, error) {
	body, err := c.Get(ctx, DetailURL(championSlug, role, opponentSlug))
	if err != nil {
		return Detail{}, err
	}
	return ParseDetail(body)
}

// ParseDetail extracts the tip text and stat rows from a matchup detail
// page's HTML.
func ParseDetail(html []byte) (Detail, error) {
	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(html))
	if err != nil {
		return Detail{}, err
	}
	if doc.Find(`label[for="championSearchAndFilter"]`).Length() == 0 {
		return Detail{}, ErrNoData
	}

	var d Detail

	// The tip is the <p> immediately following the "Tip" <strong> label.
	doc.Find("strong").EachWithBreak(func(_ int, s *goquery.Selection) bool {
		if strings.TrimSpace(s.Text()) != "Tip" {
			return true
		}
		tip := s.Parent().Next().Text()
		d.Tip = strings.TrimSpace(tip)
		return false // found it, stop iterating
	})

	// Each stat row is a div.relative.pt-7 containing exactly three spans
	// in DOM order: champion's value, the label, then the opponent's value.
	doc.Find("div.relative.pt-7").Each(func(_ int, row *goquery.Selection) {
		spans := row.Find("span")
		if spans.Length() < 3 {
			return
		}
		champVal := strings.TrimSpace(spans.Eq(0).Text())
		label := strings.TrimSpace(spans.Eq(1).Text())
		oppVal := strings.TrimSpace(spans.Eq(2).Text())
		if label == "" {
			return
		}
		d.Stats = append(d.Stats, model.Stat{
			Label:    label,
			Champion: champVal,
			Opponent: oppVal,
		})
	})

	return d, nil
}
