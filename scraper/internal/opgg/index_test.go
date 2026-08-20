package opgg

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()

	body, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %q: %v", name, err)
	}
	return body
}

func TestParseIndex(t *testing.T) {
	got, err := ParseIndex(readFixture(t, "index.html"))
	if err != nil {
		t.Fatalf("ParseIndex() error = %v", err)
	}

	want := []OpponentSummary{
		{Name: "Yasuo", WinRate: "52.31%", Games: 1234},
		{Name: "Darius", WinRate: "48.90%", Games: 987},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseIndex() = %#v, want %#v", got, want)
	}
}

func TestParseIndexNoData(t *testing.T) {
	got, err := ParseIndex(readFixture(t, "no_data.html"))
	if !errors.Is(err, ErrNoData) {
		t.Fatalf("ParseIndex() error = %v, want ErrNoData", err)
	}
	if got != nil {
		t.Fatalf("ParseIndex() = %#v, want nil", got)
	}
}
