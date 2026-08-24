// Package model defines the scraper's output unit: one matchup chunk
// per champion/opponent/role, matching the matchup_chunks Postgres schema
// described in the project brief.
package model

import (
	"fmt"
	"strings"
	"time"
)

// Stat is one labeled head-to-head number scraped from a matchup detail
// panel, e.g. {Label: "KDA", Champion: "3.08 : 1", Opponent: "1.10 : 1"}.
type Stat struct {
	Label    string `json:"label"`
	Champion string `json:"champion_value"`
	Opponent string `json:"opponent_value"`
}

// Combo is one skill-order combo card from a champion's build page, e.g.
// {Difficulty: "Medium", Sequence: "Q+AA+Q2+AA+E+AA"}. Not opponent-specific
// -- these are general combos for the champion in a role.
type Combo struct {
	Difficulty string `json:"difficulty"`
	Sequence   string `json:"sequence"`
}

// Chunk is one champion-vs-opponent-in-a-role matchup, ready to embed.
type Chunk struct {
	Champion  string    `json:"champion"`
	Opponent  string    `json:"opponent"`
	Role      string    `json:"role"`
	SourceURL string    `json:"source_url"`
	Tip       string    `json:"tip"`
	Stats     []Stat    `json:"stats"`
	ChunkText string    `json:"chunk_text"`
	CreatedAt time.Time `json:"created_at"`
}

// BuildChunkText renders the tip and stats into the flat prose block that
// gets embedded and shown to the generation model. Keeping this separate
// from parsing lets the eval harness re-render chunk_text after a prompt
// or formatting change without re-scraping.
func BuildChunkText(champion, opponent, role, tip string, stats []Stat) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s vs %s (%s lane matchup)\n", champion, opponent, capitalize(role))
	if tip != "" {
		fmt.Fprintf(&b, "Tip: %s\n", tip)
	}
	if len(stats) > 0 {
		b.WriteString("Stats:\n")
		for _, s := range stats {
			fmt.Fprintf(&b, "- %s: %s (%s) vs %s (%s)\n", s.Label, champion, s.Champion, opponent, s.Opponent)
		}
	}
	return b.String()
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
