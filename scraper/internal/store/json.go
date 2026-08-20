// Package store writes scraped chunks to disk. JSON files for now, per
// the project brief's v1 plan — swap for a Postgres writer once the
// pipeline is validated end to end.
package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"matchup-scraper/internal/model"
)

// WriteChunksJSON writes chunks to path as a single indented JSON array.
func WriteChunksJSON(path string, chunks []model.Chunk) error {
	return WriteJSON(path, chunks)
}

// WriteJSON writes any report or data value as indented JSON.
func WriteJSON(path string, value any) error {
	parent := filepath.Dir(path)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("creating output directory %s: %w", parent, err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}
