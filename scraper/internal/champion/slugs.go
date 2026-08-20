package champion

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
)

var nonAlnum = regexp.MustCompile(`[^a-z0-9]`)

// knownSlugOverrides covers the handful of champions whose op.gg URL slug
// isn't just their display name with punctuation stripped (verified
// against data/opgg_champion_slugs.json).
var knownSlugOverrides = map[string]string{
	"wukong":      "monkeyking",
	"renataglasc": "renata",
	"nunuwillump": "nunu",
}

// SlugRegistry resolves a champion's display name (as scraped from op.gg's
// own UI, e.g. an <img alt="..."> attribute) to its URL slug. It only
// returns slugs known to actually exist on op.gg, so callers can skip
// opponents it can't confidently resolve instead of guessing a URL.
type SlugRegistry struct {
	known map[string]struct{}
}

// LoadSlugRegistry reads the verified slug list at path (see
// data/opgg_champion_slugs.json).
func LoadSlugRegistry(path string) (*SlugRegistry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var slugs []string
	if err := json.Unmarshal(raw, &slugs); err != nil {
		return nil, err
	}
	known := make(map[string]struct{}, len(slugs))
	for _, s := range slugs {
		known[s] = struct{}{}
	}
	return &SlugRegistry{known: known}, nil
}

// Resolve normalizes name into a slug and returns it only if that slug is
// a known op.gg champion page. ok is false if the name couldn't be
// confidently resolved (new champion not yet in the seed list, unusual
// display name, etc.) — callers should skip rather than guess.
func (r *SlugRegistry) Resolve(name string) (slug string, ok bool) {
	normalized := nonAlnum.ReplaceAllString(strings.ToLower(name), "")
	if override, found := knownSlugOverrides[normalized]; found {
		normalized = override
	}
	_, exists := r.known[normalized]
	return normalized, exists
}
