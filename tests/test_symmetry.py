"""Symmetriegruppen der Umbenennung: Hand-Instanzen, Gültigkeit, Festpunkt, Gruppengröße und die Formeln des Other-Play-Werts."""

import math

import numpy as np
import pytest

import fok_constants as C
import fok_symmetry as SY


def test_hand_instances_of_the_two_groups(hand_relabel):
    h = hand_relabel
    assert SY.random_relabel(h["keys_all"], 3, C.GROUP_ALL).tolist() == [h["all"]]
    assert SY.random_relabel(h["keys_fix"], 3, C.GROUP_FIX0).tolist() == [h["fix"]]
    assert SY.random_relabel(np.zeros((2, 0)), 2, C.GROUP_FIX0).tolist() == [[0, 1], [0, 1]]      # m = 2: die Gruppe ist trivial


@pytest.mark.parametrize("m", [2, 3, 5, 16])
@pytest.mark.parametrize("group", [C.GROUP_ALL, C.GROUP_FIX0])
def test_relabelings_are_valid_permutations_and_fix0_keeps_mark_zero(m, group):
    rng = np.random.default_rng(m)
    phi = SY.random_relabel(rng.random((300, SY.keys_needed(m, group))), m, group)
    assert phi.shape == (300, m) and all(sorted(row) == list(range(m)) for row in phi.tolist())
    if group == C.GROUP_FIX0:
        assert (phi[:, 0] == 0).all()


def test_group_sizes_and_keys():
    assert [SY.group_size(m, C.GROUP_ALL) for m in (2, 3, 5)] == [2, 6, 120]
    assert [SY.group_size(m, C.GROUP_FIX0) for m in (2, 3, 5)] == [1, 2, 24]
    assert [SY.keys_needed(m, C.GROUP_ALL) for m in (2, 4)] == [2, 4] and [SY.keys_needed(m, C.GROUP_FIX0) for m in (2, 3, 4)] == [0, 2, 3]
    with pytest.raises(ValueError):
        SY.group_size(3, "unbekannt")


def test_every_element_of_the_group_occurs_with_equal_frequency():
    rng = np.random.default_rng(1)
    for m, group in ((3, C.GROUP_ALL), (4, C.GROUP_FIX0)):
        n = 30000
        phi = SY.random_relabel(rng.random((n, SY.keys_needed(m, group))), m, group)
        counts = {}
        for row in phi.tolist():
            counts[tuple(row)] = counts.get(tuple(row), 0) + 1
        k = SY.group_size(m, group)
        assert len(counts) == k
        se = math.sqrt(n * (1 / k) * (1 - 1 / k))
        assert all(abs(c - n / k) < 4 * se for c in counts.values())


def test_mode_to_group():
    assert SY.group_of_mode(C.MODE_SELF) is None and SY.group_of_mode(C.MODE_ALL) == C.GROUP_ALL and SY.group_of_mode(C.MODE_FIX0) == C.GROUP_FIX0
    with pytest.raises(ValueError):
        SY.group_of_mode("Quatsch")


def test_other_play_value_by_hand():
    # m = 2, p = q = (0,5; 0,5), δ = 0,2, Platz 1 fest: (1 + 0,2)·0,25 + 0,5·0,5/1 = 0,3 + 0,25 = 0,55 (bei m = 2 gleich dem Selbstspiel)
    p = np.array([0.5, 0.5])
    assert SY.other_play_value(p, p, 0.2, C.GROUP_FIX0) == pytest.approx(0.55)
    # alle Plätze, m = 4, p₀ = 0,5, δ = 1: 1/4 + 1·0,5/4 = 0,375, unabhängig von q
    p4 = np.array([0.5, 0.5, 0.0, 0.0])
    assert SY.other_play_value(p4, np.array([0.0, 0.0, 0.0, 1.0]), 1.0, C.GROUP_ALL) == pytest.approx(0.375)
    assert SY.other_play_value(p4, np.array([1.0, 0.0, 0.0, 0.0]), 1.0, C.GROUP_ALL) == pytest.approx(0.375)
    # Platz 1 fest, m = 4: p = q = (0, 1/3, 1/3, 1/3): 0 + 1·1·(1 − 0)/3 = 1/3
    q = np.array([0.0, 1 / 3, 1 / 3, 1 / 3])
    assert SY.other_play_value(q, q, 0.7, C.GROUP_FIX0) == pytest.approx(1 / 3)
