"""Konventionen: Marke eines Paares, Verteilung, Fremdspiel aus der Verteilung, Nullmodell-Streuung, Halbweg-Bonus, Kennzahlen eines Blocks."""

import math

import numpy as np
import pytest

import fok_conventions as CV
import fok_learning as L
import fok_policy as PO
import fok_scenario as S


def test_label_is_the_argmax_of_the_sum_and_ties_go_to_the_smallest_mark():
    pa = np.array([[0.6, 0.4, 0.0], [0.2, 0.5, 0.3], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    pb = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    assert CV.convention_labels(pa, pb).tolist() == [0, 1, 0, 1]


def test_counts_include_unused_marks():
    assert CV.convention_counts(np.array([0, 0, 3]), 5).tolist() == [2, 0, 0, 1, 0]


def test_cross_from_counts_by_hand_and_against_the_policies(hand_counts):
    h = hand_counts
    counts = CV.convention_counts(h["labels"], h["m"])
    assert CV.cross_from_counts(counts) == pytest.approx(h["cross"])
    pure = np.eye(h["m"])[h["labels"]]
    assert PO.cross_play(pure, pure) == pytest.approx(h["cross"])             # bei reinen Politiken ist die Vorhersage exakt
    assert CV.cross_from_counts(np.array([1, 0])) is None


def test_null_sd_by_hand_and_none_for_one_pair():
    # m = 2, P = 11: √(2·1) / (2·10) = 0,0707107
    assert CV.cross_null_sd(2, 11) == pytest.approx(math.sqrt(2) / 20)
    assert CV.cross_null_sd(4, 1) is None and CV.cross_excess_z(0.5, 4, 1) is None
    assert CV.cross_excess_z(0.25 + 0.1, 4, 101) == pytest.approx(0.1 / (math.sqrt(6) / (4 * 100)))


def test_null_sd_matches_the_spread_of_uniform_conventions():
    """Gleichverteilte Konventionen (reine Politiken): Fremdspiel − 1/m hat Mittel 0 und die Standardabweichung der Formel (Verteilungsaussage über 4 000 Stichproben)."""
    rng = np.random.default_rng(5)
    m, pairs = 4, 30
    vals = []
    for _ in range(4000):
        counts = np.bincount(rng.integers(0, m, size=pairs), minlength=m)
        vals.append(CV.cross_from_counts(counts) - 1 / m)
    vals = np.array(vals)
    assert abs(vals.mean()) < 4 * vals.std() / math.sqrt(4000)
    assert vals.std() == pytest.approx(CV.cross_null_sd(m, pairs), rel=0.1)


def test_half_way_delta_by_hand(hand_halfway):
    h = hand_halfway
    assert CV.half_way_delta(h["deltas"], h["fractions"], h["m"]) == ("interpoliert", pytest.approx(h["value"]))


def test_half_way_delta_boundary_cases():
    assert CV.half_way_delta((0.0, 0.1, 1.0), (0.5, 0.9, 0.99), 2) == ("unter", 0.1)       # schon beim ersten Bonus über dem Niveau 0,75
    assert CV.half_way_delta((0.0, 0.1, 1.0), (0.5, 0.6, 0.7), 2) == ("nie", None)
    kind, val = CV.half_way_delta((0.0, 0.1, 1.0), (0.5, 0.6, 0.75), 2)
    assert kind == "interpoliert" and val == pytest.approx(1.0)          # genau auf dem Niveau am Gitterpunkt


def test_block_metrics_on_the_hand_instances(hand_two_pairs):
    h = hand_two_pairs
    theta = np.stack([np.log(h["pa"] + 1e-300), np.log(h["pb"] + 1e-300)], axis=1)
    bm = CV.block_metrics(theta)
    assert bm["own"] == pytest.approx(1.0) and bm["cross"] == pytest.approx(0.0, abs=1e-12) and bm["frac0"] == 0.5 and bm["pred"] == pytest.approx(0.0)
    assert bm["sure"] == 1.0 and bm["pmax"] == pytest.approx(1.0)


def test_block_metrics_with_a_single_pair_have_no_cross_play():
    bm = CV.block_metrics(np.zeros((1, 2, 3)))
    assert bm["cross"] is None and bm["pred"] is None and bm["own"] == pytest.approx(1 / 3) and bm["sure"] == 0.0


def test_prediction_from_conventions_matches_the_measured_cross_play_after_training():
    """Nach dem Lernen sind die Politiken fast rein: Σ n(n−1)/(P(P−1)) liegt innerhalb von 0,02 am gemessenen Fremdspiel (gemessen etwa 0,001)."""
    bm = CV.block_metrics(L.train_pairs(S.Settings(m=4, delta=0.3, pairs=200, seed=17)).theta)
    assert abs(bm["pred"] - bm["cross"]) < 0.02
