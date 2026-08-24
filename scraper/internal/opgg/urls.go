package opgg

import (
	"fmt"
	"net/url"
)

const baseURL = "https://op.gg/lol/champions"

// IndexURL is the counters page listing every opponent's win rate and
// games for champion in role, e.g. op.gg/lol/champions/malphite/counters/top.
func IndexURL(championSlug, role string) string {
	return fmt.Sprintf("%s/%s/counters/%s", baseURL, championSlug, role)
}

// BuildURL is the champion's build page, e.g.
// op.gg/lol/champions/ambessa/build/top. The combo widget (skill-order
// button strings like "Q+AA+Q2+AA+E+AA") only appears here, client-rendered.
func BuildURL(championSlug, role string) string {
	return fmt.Sprintf("%s/%s/build/%s", baseURL, championSlug, role)
}

// DetailURL is the same page focused on one opponent via target_champion,
// which surfaces that pair's specific tip text and head-to-head stats.
func DetailURL(championSlug, role, opponentSlug string) string {
	u := fmt.Sprintf("%s/%s/counters/%s", baseURL, championSlug, role)
	q := url.Values{"target_champion": {opponentSlug}}
	return u + "?" + q.Encode()
}
