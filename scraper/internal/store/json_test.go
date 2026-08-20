package store

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"matchup-scraper/internal/model"
)

func TestWriteChunksJSONCreatesParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "chunks.json")
	want := []model.Chunk{{Champion: "Malphite", Opponent: "Yasuo", Role: "top"}}

	if err := WriteChunksJSON(path, want); err != nil {
		t.Fatalf("WriteChunksJSON() error = %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	var got []model.Chunk
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(got) != 1 || got[0].Champion != want[0].Champion || got[0].Opponent != want[0].Opponent {
		t.Fatalf("decoded chunks = %#v, want %#v", got, want)
	}
}
