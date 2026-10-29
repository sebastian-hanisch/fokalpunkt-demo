"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Zahlen, Reglerränder, Entartungen, Überläufe, Division durch null."""

import math

import numpy as np
import pytest

import fok_constants as C
import fok_conventions as CV
import fok_evaluation as V
import fok_experience as EX
import fok_experiments as XP
import fok_learning as L
import fok_policy as PO
import fok_scenario as S
import fok_symmetry as SY


def test_exactly_two_places_is_the_smallest_game():
    """m = 2: Softmax, Ziehung, Other-Play (trivial), Halbweg-Niveau (0,75) und Gleichstand der Konvention greifen nicht auf einen Zweig `m < 2` zurück."""
    s = S.Settings(m=2, pairs=10, steps=50, delta=0.5, seed=3)
    assert np.isfinite(L.train_pairs(s).theta).all()
    for mode in C.MODES:
        assert np.isfinite(L.train_pairs(S.with_(s, mode=mode)).theta).all()
    assert CV.half_way_delta((0.0, 0.1, 1.0), (0.5, 0.8, 0.9), 2) == ("unter", 0.1)
    assert SY.other_play_value(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.0, C.GROUP_FIX0) == pytest.approx(0.5)


def test_exactly_two_pairs_is_the_smallest_sample_with_a_stranger():
    p = PO.softmax(L.train_pairs(S.Settings(pairs=2, steps=100)).theta)
    cross = PO.cross_play(p[:, 0], p[:, 1])
    assert 0.0 <= cross <= 1.0 and CV.cross_null_sd(4, 2) == pytest.approx(math.sqrt(6) / 4)
    assert CV.cross_from_counts(np.array([2, 0, 0, 0])) == 1.0 and CV.cross_from_counts(np.array([1, 1, 0, 0])) == 0.0


def test_one_pair_has_no_stranger_and_nothing_divides_by_zero():
    v = V.analyse(S.Settings(pairs=1, steps=200))
    assert v.cross is None and v.pred is None and np.isfinite(v.own) and v.matrix.shape == (1, 1)
    kind, text = V.verdict_cross(v)
    assert kind == "info" and "keinen Fremden" in text
    v = V.analyse(S.Settings(pairs=1, steps=200, t0=50))
    assert v.cross_within is None and v.cross_between is None


def test_two_pairs_with_common_history_form_one_group_without_between_play():
    v = V.analyse(S.Settings(pairs=2, steps=200, t0=50))
    assert v.group.tolist() == [0, 0] and v.cross_between is None and v.cross_within is not None


def test_three_pairs_with_history_form_one_group_so_every_group_has_a_stranger():
    v = V.analyse(S.Settings(pairs=3, steps=100, t0=25))
    assert v.group.tolist() == [0, 0, 0] and v.cross_within is not None


def test_one_step_and_the_smallest_slider_values_run():
    s = S.Settings(m=C.M_MIN, delta=C.DELTA_MIN, eta=C.ETA_MIN, sigma0=C.SIGMA_MIN, steps=1, pairs=C.PAIRS_MIN, t0=C.T0_MIN)
    v = V.analyse(s)
    assert np.isfinite(v.own) and v.own == pytest.approx(0.5, abs=0.01)             # σ₀ = 0 und ein Schritt: höchstens eine kleine Verschiebung bei einem Treffer
    assert np.isfinite(L.train_pairs(S.with_(s, steps=C.STEPS_MIN, pairs=2)).theta).all()


def test_largest_slider_values_stay_finite_and_inside_the_budget():
    s = S.Settings(m=C.M_MAX, delta=C.DELTA_MAX, eta=C.ETA_MAX, sigma0=C.SIGMA_MAX, steps=C.STEPS_MAX, pairs=C.PAIRS_MAX, t0=C.T0_MAX, mode=C.MODE_FIX0)
    assert S.budget_pairs(s) == C.RUN_BUDGET // C.STEPS_MAX and S.budget_pairs(s) < C.PAIRS_MAX
    small = S.with_(s, pairs=4, steps=3000, t0=300)
    v = V.analyse(small)
    assert all(np.isfinite(x) for x in (v.own, v.cross, v.pmax, v.sure)) and np.isfinite(v.trace).all() and np.isfinite(v.matrix).all()


def test_budget_never_goes_below_one_pair():
    assert S.budget_pairs(S.Settings(pairs=1, steps=C.STEPS_MAX)) == 1
    assert S.estimated_seconds(S.Settings()) > 0


def test_no_overflow_in_softmax_or_sampling_for_extreme_parameters_after_long_training():
    """Nach vielen Schritten mit großer Lernrate werden Parameter groß; Softmax zieht das Maximum ab, nichts wird NaN."""
    s = S.Settings(m=16, delta=3.0, eta=0.3, pairs=5, steps=3000, sigma0=0.5)
    tr = L.train_pairs(s)
    p = PO.softmax(tr.theta)
    assert np.isfinite(tr.theta).all() and np.isfinite(p).all() and p.sum(axis=-1) == pytest.approx(np.ones((5, 2)), abs=1e-12)
    huge = np.full((3, 2, 4), 0.0)
    huge[:, :, 0] = 1e6
    assert np.isfinite(PO.softmax(huge)).all() and PO.sample_actions(PO.softmax(huge), np.full((3, 2), 0.7)).tolist() == [[0, 0]] * 3


def test_all_equal_parameters_start_uniform_and_ties_in_the_convention_go_to_the_smallest_mark():
    th = np.zeros((4, 2, 5))
    p = PO.softmax(th)
    assert (CV.convention_labels(p[:, 0], p[:, 1]) == 0).all()
    assert V.analyse(S.Settings(pairs=4, steps=0, sigma0=0.0)).frac0 == 1.0            # kein Lernen, gleichverteilt: die Gleichstände gehen an Platz 1 (festgelegt, kein Befund)


def test_zero_bonus_gives_the_same_reward_on_every_place():
    a = np.arange(6)
    assert L.payoff(a, a, 0.0).tolist() == [1.0] * 6 and L.payoff(a, a, 2.0).tolist() == [3.0] + [1.0] * 5


def test_community_with_two_players_and_one_community():
    out = EX.community(3, 2, 1, 30, 0.1, 0.0, 0.1, L.Stream(1), (30,))
    assert out[30]["between"] is None and 0.0 <= out[30]["within"] <= 1.0


def test_reference_with_zero_history_steps_and_one_pair_per_group():
    c = XP.reference_cell(S.Settings(m=3, steps=30), 3, 1, 0, 2, "randfall|ref")
    assert c["within"]["mean"] is None and c["between"]["mean"] is not None               # ein Paar je Gruppe: kein Innenspiel


def test_cell_with_a_single_pair_per_block_has_no_cross_play_and_no_crash():
    c = XP.cell(S.Settings(pairs=1, steps=20), 2, "randfall|ein paar")
    assert c["cross"]["mean"] is None and c["own"]["mean"] is not None and c["pairs"] == 2


def test_stats_of_one_block_has_zero_error():
    assert XP.stats([{"a": 0.3}])["a"] == {"mean": 0.3, "se": 0.0}


def test_half_way_never_reached_and_fractions_equal_to_the_level():
    assert CV.half_way_delta((0.0, 0.1, 1.0), (0.25, 0.3, 0.4), 4) == ("nie", None)
    kind, val = CV.half_way_delta((0.0, 0.1, 1.0), (0.25, 0.3, 0.625), 4)
    assert kind == "interpoliert" and val == pytest.approx(1.0)


def test_permalink_style_limits_exist_for_every_setting():
    assert C.M_MIN < C.M_MAX and C.DELTA_MIN < C.DELTA_MAX and C.STEPS_MIN < C.STEPS_MAX and C.PAIRS_MIN < C.PAIRS_MAX and C.T0_MIN < C.T0_MAX and C.ETA_MIN < C.ETA_MAX and C.SIGMA_MIN < C.SIGMA_MAX
