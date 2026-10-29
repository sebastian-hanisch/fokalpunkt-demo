"""Geteilte Vorerfahrung und Gemeinschaften: Gruppenbildung, gemeinsamer Start, Innen- gegen Zwischenspiel, Gemeinschaftslernen."""

import numpy as np
import pytest

import fok_experience as EX
import fok_learning as L
import fok_policy as PO
import fok_scenario as S


def test_group_index_by_hand():
    assert EX.group_index(6, 3).tolist() == [0, 0, 1, 1, 2, 2]
    assert EX.group_index(5, 2).tolist() == [0, 0, 0, 1, 1]
    assert EX.group_index(1, 1).tolist() == [0]


def test_ref_count_keeps_at_least_two_pairs_per_group():
    assert [EX.ref_count(p) for p in (1, 2, 3, 4, 10, 16, 17, 500)] == [1, 1, 1, 2, 5, 8, 8, 8]
    for pairs in range(2, 60):
        sizes = np.bincount(EX.group_index(pairs, EX.ref_count(pairs)))
        assert sizes.min() >= 2 and sizes.sum() == pairs


def test_shared_reference_gives_every_pair_of_a_group_the_same_start():
    s = S.Settings(m=4, t0=50, seed=3)
    start, group = EX.shared_reference(s, 3, 5, L.Stream(3))
    assert start.shape == (15, 2, 4) and group.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    for g in range(3):
        block = start[group == g]
        assert (block == block[0]).all()
    assert not np.array_equal(start[0], start[5])


def test_shared_reference_with_zero_steps_is_the_common_random_start():
    s = S.Settings(m=3, t0=0, seed=4)
    start, _ = EX.shared_reference(s, 2, 3, L.Stream(4))
    ref = L.initial_theta(2, 3, s.sigma0, L.Stream(4))
    assert np.array_equal(start[0], ref[0]) and np.array_equal(start[3], ref[1])


def test_groups_for_pairs_in_the_main_view():
    s = S.Settings(m=4, t0=30, pairs=20, seed=2)
    theta0, group = EX.groups_for_pairs(s, 20, L.Stream(2))
    assert theta0.shape == (20, 2, 4) and len(np.bincount(group)) == EX.ref_count(20) and np.bincount(group).min() >= 2
    assert (theta0[group == 0] == theta0[group == 0][0]).all()


def test_common_history_lifts_cross_play_within_a_group_above_between_groups():
    """Zweig-Test: eine gemeinsame Vorgeschichte hebt das Fremdspiel innerhalb der Gruppe deutlich über das zwischen den Gruppen (m = 4, T₀ = 400, 8 Referenzen mit je 20 Paaren)."""
    s = S.Settings(m=4, t0=400, steps=3000, seed=9)
    stream = L.Stream(9)
    start, group = EX.shared_reference(s, 8, 20, stream)
    p = PO.softmax(L.train_pairs(s, theta0=start, stream=stream, pairs=160).theta)
    within, between = PO.cross_play_groups(p[:, 0], p[:, 1], group)
    assert within > 0.8 and between < 0.6


def test_community_rejects_an_odd_or_tiny_size():
    for size in (1, 3):
        with pytest.raises(ValueError):
            EX.community(4, size, 2, 5, 0.1, 0.0, 0.1, L.Stream(1), (5,))


def test_community_reports_each_checkpoint_with_bounded_values():
    out = EX.community(4, 4, 6, 40, 0.1, 0.0, 0.1, L.Stream(2), (10, 40))
    assert set(out) == {10, 40}
    for v in out.values():
        assert 0.0 <= v["within"] <= 1.0 and 0.0 <= v["sure"] <= 1.0 and 0.0 <= v["between"] <= 1.0


def test_community_stats_on_a_hand_instance():
    # zwei Gemeinschaften zu je zwei Spielern: Gemeinschaft 0 beide auf Marke 0, Gemeinschaft 1 beide auf Marke 1 (m = 2)
    p = np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    st = EX.community_stats(p)
    assert st == {"within": 1.0, "sure": 1.0, "between": 0.0}
    assert EX.community_stats(p[:1])["between"] is None             # eine Gemeinschaft: kein Zwischenspiel


def test_communities_settle_within_and_stay_at_chance_between_in_distribution():
    """Verteilungsaussage über 100 Gemeinschaften mit je 4 Spielern nach 1 500 Runden: innen deutlich über, zwischen bei 1/m (Marge 0,05)."""
    out = EX.community(4, 4, 100, 1500, 0.1, 0.0, 0.1, L.Stream(5), (1500,))[1500]
    assert out["within"] > 0.9 and abs(out["between"] - 0.25) < 0.05
