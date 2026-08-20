package opgg

import (
	"errors"
	"reflect"
	"testing"

	"matchup-scraper/internal/model"
)

func TestParseDetail(t *testing.T) {
	got, err := ParseDetail(readFixture(t, "detail.html"))
	if err != nil {
		t.Fatalf("ParseDetail() error = %v", err)
	}

	want := Detail{
		Tip: "Use your armor advantage early and save your crowd control\n        for the opponent's engage.",
		Stats: []model.Stat{
			{Label: "Win Rate", Champion: "53.2%", Opponent: "46.8%"},
			{Label: "KDA", Champion: "3.08 : 1", Opponent: "1.10 : 1"},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseDetail() = %#v, want %#v", got, want)
	}
}

func TestParseDetailNoData(t *testing.T) {
	got, err := ParseDetail(readFixture(t, "no_data.html"))
	if !errors.Is(err, ErrNoData) {
		t.Fatalf("ParseDetail() error = %v, want ErrNoData", err)
	}
	if !reflect.DeepEqual(got, Detail{}) {
		t.Fatalf("ParseDetail() = %#v, want zero Detail", got)
	}
}
