import json, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
import report  # noqa: E402

LABELS = ["card_arrival", "card_not_working", "lost_or_stolen_card", "top_up_failed", "Refund_not_showing_up"]


@pytest.mark.parametrize("raw,expected", [
    ("card_arrival", "card_arrival"),
    ("Label: card arrival", "card_arrival"),
    ("`lost_or_stolen_card`.", "lost_or_stolen_card"),
    ("refund not showing up", "Refund_not_showing_up"),
    ("I think the intent is top_up_failed because...", "top_up_failed"),
    ("The answer is card_not_working", "card_not_working"),
    ("pizza", "INVALID"),
    ("", "INVALID"),
    (None, "INVALID"),
])
def test_parse_label(raw, expected):
    assert common.parse_label(raw, LABELS) == expected


def test_parse_label_prefers_longest_match():
    labels = ["card", "card_not_working"]
    assert common.parse_label("it's card not working", labels) == "card_not_working"


def test_scores_macro_f1_and_invalid_rate():
    s = common.scores(["a", "a", "b", "b"], ["a", "a", "b", "INVALID"])
    assert s["accuracy"] == 0.75 and s["invalid_rate"] == 0.25
    # class a: P=1, R=1 -> 1.0 ; class b: P=1, R=0.5 -> 0.667 ; macro = 0.8333
    assert s["macro_f1"] == pytest.approx(0.8333, abs=1e-4)


def test_cost_per_1k_self_hosted_and_api():
    # 1000 predictions x 3.6 s at $1/h = 1 hour = $1 per 1k
    assert common.cost_per_1k([3.6] * 1000, 1.0, 0, 0, None, None) == pytest.approx(1.0)
    # API: 1M in tokens at $1 + 0.5M out at $4 over 1000 predictions = $3 per 1k
    assert common.cost_per_1k([1] * 1000, None, 1_000_000, 500_000, 1.0, 4.0) == pytest.approx(3.0)
    assert common.cost_per_1k([1], None, 10, 10, None, None) is None


def test_percentile():
    assert common.percentile([], 0.95) == 0.0
    assert common.percentile(list(range(101)), 0.95) == 95


def test_report_latest_row_wins_and_sorted(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [{"run": "x", "macro_f1": 0.1, "n": 1}, {"run": "y", "macro_f1": 0.5, "n": 1}, {"run": "x", "macro_f1": 0.9, "n": 1}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = report.load(p)
    assert {r["run"]: r["macro_f1"] for r in loaded} == {"x": 0.9, "y": 0.5}
    md = report.table(loaded)
    assert md.index("| x |") < md.index("| y |")
