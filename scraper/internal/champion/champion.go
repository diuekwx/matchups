// Package champion holds the seed roster the scraper crawls: which
// champions to pull matchup data for, and in which roles.
package champion

import (
	"encoding/json"
	"fmt"
	"os"
)

// Role is a lane/position on op.gg's counters pages.
type Role string

const (
	Top     Role = "top"
	Jungle  Role = "jungle"
	Mid     Role = "mid"
	ADC     Role = "adc"
	Support Role = "support"
)

// Seed is one champion and the roles to crawl matchups for.
type Seed struct {
	Slug  string `json:"slug"`  // op.gg URL slug, e.g. "monkeyking" for Wukong
	Name  string `json:"name"`  // display name, e.g. "Wukong"
	Roles []Role `json:"roles"` // roles this champion is played in
}

// LoadSeeds reads the champion roster from a JSON file (see data/champions.json).
func LoadSeeds(path string) ([]Seed, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading champion seed file: %w", err)
	}
	var seeds []Seed
	if err := json.Unmarshal(raw, &seeds); err != nil {
		return nil, fmt.Errorf("parsing champion seed file: %w", err)
	}
	return seeds, nil
}
