"""Einstellungen und die Live-Fassungen der Experimente (kleine Zellen): Form, Reproduzierbarkeit, Zählregeln."""

import numpy as np
import pytest

import fok_constants as C
import fok_experiments as XP
import fok_scenario as S


def test_settings_must_be_built_with_keywords():
    with pytest.raises(TypeError):
        S.Settings(4, 0.0)          # positionale Argumente sind verboten (ein eingefügtes Feld würde Werte still verschieben)


def test_settings_are_frozen_and_with_changes_only_the_named_fields():
    s = S.Settings(m=5)
    with pytest.raises(Exception):
        s.m = 6
    t = S.with_(s, delta=0.5)
    assert (t.m, t.delta, t.pairs) == (5, 0.5, C.DEFAULT_PAIRS) and s.delta == 0.0


def test_budget_and_time_estimate():
    assert S.budget_pairs(S.Settings(pairs=500, steps=6000)) == 500 and S.budget_pairs(S.Settings(pairs=500, steps=20000)) == 150
    assert S.estimated_seconds(S.Settings(pairs=500, steps=6000, m=4)) == pytest.approx(500 * 6000 / C.PAIR_STEPS_PER_SECOND)
    assert S.estimated_seconds(S.Settings(m=16)) == pytest.approx(2 * S.estimated_seconds(S.Settings(m=4)))


def test_seed_of_is_a_stable_fnv_hash_and_inside_the_seed_range():
    assert XP.seed_of("e1|4|0.0") == XP.seed_of("e1|4|0.0") != XP.seed_of("e1|4|0.1")
    assert all(0 <= XP.seed_of(f"x{i}") <= C.SEED_MAX for i in range(50))
    assert XP.seed_of("") == 0xCBF29CE484222325 % (C.SEED_MAX + 1)


def test_stats_mean_and_standard_error():
    out = XP.stats([{"a": 0.2}, {"a": 0.4}, {"a": 0.6}, {"a": None}][:3])
    assert out["a"]["mean"] == pytest.approx(0.4) and out["a"]["se"] == pytest.approx(0.2 / 3 ** 0.5) and out["blocks"] == 3
    assert XP.stats([{"a": 0.2}, {"a": None}])["a"] == {"mean": None, "se": None}


def test_a_cell_is_reproducible_and_has_the_documented_keys():
    s = S.Settings(m=3, pairs=20, steps=200)
    a, b = XP.cell(s, 2, "test|cell"), XP.cell(s, 2, "test|cell")
    assert a == b and set(a) == {"blocks", "own", "cross", "frac0", "pred", "sure", "pmax", "pairs"} and a["pairs"] == 40
    assert XP.cell(s, 2, "test|andere")["cross"]["mean"] != a["cross"]["mean"]


def test_delta_row_covers_the_grid_in_order():
    row = XP.delta_row(S.Settings(m=2, pairs=10, steps=50), 1, "test|row", deltas=(0.0, 0.5, 2.0))
    assert list(row) == [0.0, 0.5, 2.0]


def test_e1_live_returns_every_m_and_the_three_deltas():
    out = XP.e1_live(S.Settings(pairs=8, steps=60))
    assert list(out) == list(C.MS) and all(list(r) == [0.0, 0.1, 1.0] for r in out.values())


def test_flow_cell_keys_and_the_start_favourite_in_the_flow_without_bonus():
    c = XP.flow_cell(S.Settings(m=3, pairs=60, steps=1500, delta=0.0), 1, "test|flow")
    assert set(c) == {"blocks", "det0", "sto0", "init0", "agree", "init_det", "init_sto", "det_own", "pairs"}
    assert c["init_det"]["mean"] > 0.9            # ohne Bonus gewinnt im Fluss der Start-Favorit (Verteilungsaussage über 60 Paare)


def test_reference_cell_with_a_long_history_has_high_inner_cross_play_and_chance_between():
    c = XP.reference_cell(S.Settings(m=2, steps=1000), 10, 8, 600, 1, "test|ref")
    assert c["within"]["mean"] > 0.9 and abs(c["between"]["mean"] - 0.5) < 0.35 and c["pairs"] == 80


def test_e4_live_uses_the_live_sizes():
    out = XP.e4_live(S.Settings(m=2, steps=100), t0s=(0, 50), refs=3, per_ref=2)
    assert list(out) == [0, 50] and out[0]["pairs"] == 6


def test_community_cell_structure():
    c = XP.community_cell(4, 2, 4, 2, "test|comm", checkpoints=(10, 20))
    assert set(c) == {"size", "communities", "10", "20"} and c["communities"] == 8 and set(c["10"]) == {"blocks", "within", "sure", "between"}


def test_small_sample_structure_and_group_split():
    c = XP.small_sample(2, 0.2, 5, 100, 6, "test|small")
    assert len(c["values"]) == 6 and c["pairs_per"] == 5 and c["groups"] == 6 and c["ge81"] == sum(1 for v in c["values"] if v >= 0.81)


def test_run_job_dispatches_every_kind():
    kinds = {j[0] for j in XP.jobs()}
    assert kinds == {"cell", "flow", "ref", "small", "comm"}
    with pytest.raises(ValueError):
        XP.run_job(("unbekannt", ("x",)))


def test_to_jsonable_converts_numpy_and_keys():
    out = XP.to_jsonable({0.5: np.float64(1.5), 3: [np.int64(2), (1, 2)]})
    assert out == {"0.5": 1.5, "3": [2, [1, 2]]} and type(out["0.5"]) is float
