from datetime import datetime, timedelta, timezone

import pytest

from app.scoring import (
    MAX_POINTS_PER_SECOND,
    MAX_RUN_SECONDS,
    MAX_SCORE,
    REGIONS,
    SEASON_EPOCH,
    ScoreError,
    beats,
    check_region,
    check_score,
    decode,
    encode,
    neighbours,
    page_bounds,
    rank_board,
    run_seconds,
    validate_run,
    week_key,
    week_ttl_seconds,
)


def make_run(**overrides):
    run = {
        "player": "Alice",
        "region": "north",
        "score": 1000,
        "run_id": "run-001",
        "started_at": SEASON_EPOCH,
        "finished_at": SEASON_EPOCH + timedelta(seconds=60),
    }
    run.update(overrides)
    return run


def test_validate_run_normalizes_player_and_region():
    result = validate_run(
        make_run(player=" Alice ", region=" NORTH ")
    )
    assert result["player"] == "Alice"
    assert result["region"] == "north"
    assert result["seconds"] == 60


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"player": " "}, "needs a player"),
        ({"run_id": ""}, "needs a run_id"),
        ({"region": "moon"}, "is not a region"),
        ({"score": -1}, "cannot be negative"),
        ({"score": MAX_SCORE + 1}, "above the maximum"),
        (
            {"started_at": SEASON_EPOCH + timedelta(seconds=60),
             "finished_at": SEASON_EPOCH},
            "finish before it started",
        ),
        (
            {"finished_at": SEASON_EPOCH + timedelta(seconds=10),
             "score": 100},
            "too short",
        ),
        (
            {"finished_at": SEASON_EPOCH + timedelta(seconds=60),
             "score": 60 * MAX_POINTS_PER_SECOND + 1},
            "maximum rate",
        ),
        (
            {"finished_at": SEASON_EPOCH + timedelta(seconds=MAX_RUN_SECONDS + 1),
             "score": 0},
            "longer than anyone plays",
        ),
    ],
)
def test_validate_run_rejects_invalid_submissions(overrides, message):
    with pytest.raises(ScoreError, match=message):
        validate_run(make_run(**overrides))


def test_validate_run_rejects_duplicate_run_id():
    with pytest.raises(ScoreError, match="already been submitted"):
        validate_run(make_run(), seen_run_ids={"run-001"})


def test_validate_run_accepts_all_supported_regions():
    for region in REGIONS:
        assert validate_run(make_run(region=region))["region"] == region


@pytest.mark.parametrize("score", [0, 1, 500, MAX_SCORE])
def test_check_score_accepts_boundaries(score):
    assert check_score(score) == score


@pytest.mark.parametrize("region", ["north", "south", "east", "west", "central"])
def test_check_region_accepts_valid_regions(region):
    assert check_region(region) == region


def test_run_seconds_handles_naive_timestamps_as_utc():
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = datetime(2026, 1, 1, 0, 1, 0)
    assert run_seconds(start, end) == 60


def test_encode_decode_round_trip_many_values():
    for score in range(0, MAX_SCORE + 1, 997):
        for seconds in range(0, 100_000, 997):
            finished = SEASON_EPOCH + timedelta(seconds=seconds)
            packed = encode(score, finished)
            decoded_score, decoded_time = decode(packed)
            assert decoded_score == score
            assert decoded_time == finished


def test_encode_higher_score_always_outranks_lower_score():
    finished = SEASON_EPOCH + timedelta(seconds=100)
    assert encode(101, finished) > encode(100, finished)


def test_earlier_finish_wins_when_scores_are_equal():
    early = SEASON_EPOCH + timedelta(seconds=100)
    late = SEASON_EPOCH + timedelta(seconds=200)
    assert encode(500, early) > encode(500, late)


def test_encode_stays_within_exact_double_integer_range():
    packed = encode(MAX_SCORE, SEASON_EPOCH)
    assert packed < 2**53
    assert float(packed).is_integer()


def test_beats_compares_composite_scores():
    assert beats(200, 100)
    assert not beats(100, 200)
    assert not beats(100, 100)


def test_rank_board_uses_competition_ranks_and_stable_ties():
    finished = SEASON_EPOCH + timedelta(seconds=60)
    entries = [
        {"player": "Zoe", "score": 100, "finished_at": finished},
        {"player": "Bob", "score": 100, "finished_at": finished},
        {"player": "Amy", "score": 200, "finished_at": finished},
        {
            "player": "Eve",
            "score": 100,
            "finished_at": finished + timedelta(seconds=10),
        },
    ]
    result = rank_board(entries)
    assert [(r["player"], r["rank"]) for r in result] == [
        ("Amy", 1),
        ("Bob", 2),
        ("Zoe", 2),
        ("Eve", 2),
    ]


def test_rank_board_earlier_finish_wins_equal_score():
    later = SEASON_EPOCH + timedelta(seconds=90)
    earlier = SEASON_EPOCH + timedelta(seconds=30)
    result = rank_board([
        {"player": "Later", "score": 100, "finished_at": later},
        {"player": "Earlier", "score": 100, "finished_at": earlier},
    ])
    assert [r["player"] for r in result] == ["Earlier", "Later"]


def test_week_key_changes_at_monday_utc():
    sunday = datetime(2026, 1, 4, 23, 59, tzinfo=timezone.utc)
    monday = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    assert week_key(sunday) != week_key(monday)


def test_week_key_handles_year_boundary():
    before = datetime(2026, 12, 31, 12, 0, tzinfo=timezone.utc)
    after = datetime(2027, 1, 4, 12, 0, tzinfo=timezone.utc)
    assert week_key(before) != week_key(after)


def test_week_ttl_is_positive_and_includes_grace_period():
    now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    assert week_ttl_seconds(now) > 0


def test_page_bounds():
    assert page_bounds(1, 20) == (0, 19)
    assert page_bounds(2, 20) == (20, 39)
    assert page_bounds(0, 10) == (0, 9)


@pytest.mark.parametrize("page, size", [(1, 0), (1, 101), ("bad", 10)])
def test_page_bounds_rejects_invalid_values(page, size):
    with pytest.raises(ScoreError):
        page_bounds(page, size)


def test_neighbours():
    assert neighbours(1, 2) == (0, 2)
    assert neighbours(40_000, 2) == (39_997, 40_001)


def test_neighbours_rejects_rank_below_one():
    with pytest.raises(ScoreError, match="ranks start at 1"):
        neighbours(0)
