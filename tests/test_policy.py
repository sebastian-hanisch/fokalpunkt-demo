"""Politiken: Softmax, Ziehung aus der Gleichverteilten, Erfolgswahrscheinlichkeit, Eigen- und Fremdspiel (Mini-Instanzen von Hand, Rand- und Entartungsfälle)."""

import numpy as np
import pytest

import fok_policy as PO


def test_match_probability_on_the_hand_instances(hand_match):
    for p, q, want in hand_match:
        assert PO.match_probability(np.array(p), np.array(q)) == pytest.approx(want, abs=1e-15)


def test_softmax_by_hand_and_shift_invariance(hand_softmax):
    assert PO.softmax(np.array([0.0, hand_softmax["ln2"]])) == pytest.approx([1 / 3, 2 / 3], abs=1e-15)
    assert PO.softmax(np.zeros(3)) == pytest.approx([1 / 3] * 3, abs=1e-15)
    assert PO.softmax(np.array([1000.0, 1000.0 + hand_softmax["ln2"]])) == pytest.approx([1 / 3, 2 / 3], abs=1e-12)


def test_softmax_does_not_overflow_for_huge_parameters():
    p = PO.softmax(np.array([[1e5, 0.0, -1e5], [-1e9, -1e9, 1e9]]))
    assert np.isfinite(p).all() and p[0].tolist() == [1.0, 0.0, 0.0] and p[1].tolist() == [0.0, 0.0, 1.0]
    assert PO.softmax(np.zeros((4, 16))).sum(axis=-1) == pytest.approx(np.ones(4), abs=1e-15)


def test_sampling_follows_the_cumulative_distribution_with_exact_boundaries():
    p = np.tile([0.5, 0.5], (4, 1))
    # u = 0,49 → 0; u = 0,5 liegt genau auf der Grenze und geht zur nächsten Marke → 1; u = 0,51 → 1; u nahe 1 → 1
    assert PO.sample_actions(p, np.array([0.49, 0.5, 0.51, 0.9999999])).tolist() == [0, 1, 1, 1]
    assert PO.sample_actions(np.array([[0.2, 0.3, 0.5]]), np.array([0.19])).tolist() == [0]
    assert PO.sample_actions(np.array([[0.2, 0.3, 0.5]]), np.array([0.21])).tolist() == [1]
    assert PO.sample_actions(np.array([[0.2, 0.3, 0.5]]), np.array([0.51])).tolist() == [2]


def test_sampling_never_returns_a_mark_beyond_the_last_even_if_the_sum_rounds_below_one():
    p = np.full((1, 7), 1 / 7)             # die Summe von sieben 1/7 ist nicht genau 1
    assert PO.sample_actions(p, np.array([1.0 - 2 ** -53])).tolist() == [6]
    assert PO.sample_actions(p, np.array([1.5])).tolist() == [6]


def test_sampling_a_pure_policy_always_gives_its_mark():
    p = np.tile([0.0, 0.0, 1.0, 0.0], (5, 1))
    assert PO.sample_actions(p, np.array([0.0, 0.25, 0.5, 0.75, 0.999])).tolist() == [2] * 5


def test_two_pure_pairs_with_different_conventions(hand_two_pairs):
    h = hand_two_pairs
    assert PO.own_play(h["pa"], h["pb"]) == h["own"]
    assert PO.cross_play(h["pa"], h["pb"]) == h["cross"]


def test_cross_play_needs_two_pairs_and_does_not_divide_by_zero():
    pa = np.array([[0.5, 0.5]])
    assert PO.cross_play(pa, pa) is None
    assert PO.own_play(pa, pa) == pytest.approx(0.5)


def test_cross_play_excludes_the_diagonal():
    pa = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Paare i ≠ j: (0,1) trifft 1, (1,0) 1, (0,2) 0, (2,0) 0, (1,2) 0, (2,1) 0 → 2/6
    assert PO.cross_play(pa, pa) == pytest.approx(2 / 6)


def test_group_cross_play_splits_within_and_between():
    pa = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    group = np.array([0, 0, 1, 1])
    assert PO.cross_play_groups(pa, pa, group) == (1.0, 0.0)
    assert PO.cross_play_groups(pa[:2], pa[:2], np.array([0, 0])) == (1.0, None)       # nur eine Gruppe: kein Zwischenspiel
    assert PO.cross_play_groups(pa[:2], pa[:2], np.array([0, 1])) == (None, 1.0)       # je ein Paar je Gruppe: kein Innenspiel


def test_success_matrix_diagonal_is_the_own_play_of_each_pair():
    rng = np.random.default_rng(3)
    pa, pb = PO.softmax(rng.normal(size=(6, 5))), PO.softmax(rng.normal(size=(6, 5)))
    s = PO.success_matrix(pa, pb)
    assert np.diag(s) == pytest.approx((pa * pb).sum(axis=1), abs=1e-15) and ((s >= 0) & (s <= 1)).all()


def test_a_mark_with_probability_zero_is_never_drawn_even_for_u_exactly_zero():
    """Gefundener Randfall: mit `u > kumuliert` wäre bei u = 0 die Marke 0 gezogen worden, obwohl ihre Wahrscheinlichkeit 0 ist."""
    p = np.array([[0.0, 0.5, 0.0, 0.5]])
    assert [PO.sample_actions(p, np.array([u])).tolist()[0] for u in (0.0, 0.4999, 0.5, 0.9999)] == [1, 1, 3, 3]
