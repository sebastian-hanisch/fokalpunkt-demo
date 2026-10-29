"""Lernen: Belohnung, REINFORCE-Schritt, Zufallsstrom, Training (Reproduzierbarkeit, Blockgröße, Aufzeichnung), Erwartungs-Update und -Fluss.

Simulationen nur mit festen SplitMix64-Seeds und Verteilungsaussagen über viele Paare mit Marge (Softmax-REINFORCE über tausende Schritte ist chaotisch: nichts an einem einzelnen Paar)."""

import numpy as np
import pytest

import fok_constants as C
import fok_learning as L
import fok_policy as PO
import fok_scenario as S
from fok_rng import uniform_block


def test_payoff_table():
    a, b = np.array([0, 0, 1, 2, 3]), np.array([0, 1, 1, 1, 3])
    assert L.payoff(a, b, 0.5).tolist() == [1.5, 0.0, 1.0, 0.0, 1.0]
    assert L.payoff(a, b, 0.0).tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_reinforce_step_by_hand(hand_step):
    h = hand_step
    out = L.reinforce_step(h["theta"], PO.softmax(h["theta"]), h["a"], h["reward"], h["eta"])
    assert out[0, 0].tolist() == pytest.approx(list(h["delta"]), abs=1e-15)
    assert out[0, 1].tolist() == pytest.approx(list(h["delta"]), abs=1e-15)          # beide Spieler haben Marke 0 gewählt


def test_reinforce_step_changes_nothing_without_a_hit_and_keeps_the_sum_of_parameters():
    rng = np.random.default_rng(2)
    theta = rng.normal(size=(5, 2, 4))
    p = PO.softmax(theta)
    a = rng.integers(0, 4, size=(5, 2))
    assert np.array_equal(L.reinforce_step(theta, p, a, np.zeros(5), 0.1), theta)
    out = L.reinforce_step(theta, p, a, np.ones(5), 0.1)
    assert (out - theta).sum(axis=-1) == pytest.approx(np.zeros((5, 2)), abs=1e-15)          # e_a − p hat die Summe 0: der Mittelwert der Parameter bleibt


def test_stream_blocks_concatenate_to_the_plain_block():
    st = L.Stream(7)
    joined = np.concatenate([st.take(5), st.take(11), st.take(1)])
    assert np.array_equal(joined, uniform_block(7, 17))


def test_normals_are_roughly_standard_normal_and_finite():
    z = L.normals(L.Stream(11), 100_000)
    assert np.isfinite(z).all() and abs(z.mean()) < 4 / 100_000 ** 0.5 and abs(z.std() - 1) < 4 * (0.5 / 100_000) ** 0.5 * 1.5


def test_initial_theta_scale_and_zero_sigma():
    th = L.initial_theta(2000, 4, 0.1, L.Stream(3))
    assert th.shape == (2000, 2, 4) and abs(th.std() - 0.1) < 0.005
    assert (L.initial_theta(5, 4, 0.0, L.Stream(3)) == 0.0).all()


def test_training_is_reproducible_and_depends_on_the_seed():
    s = S.Settings(m=4, pairs=20, steps=300, seed=5)
    a, b = L.train_pairs(s).theta, L.train_pairs(s).theta
    assert np.array_equal(a, b)
    assert not np.array_equal(a, L.train_pairs(S.with_(s, seed=6)).theta)


def test_block_size_does_not_change_the_numbers(monkeypatch):
    s = S.Settings(m=3, pairs=7, steps=250, seed=9, mode=C.MODE_FIX0)
    ref = L.train_pairs(s).theta
    monkeypatch.setattr(C, "CHUNK", 13)
    assert np.array_equal(L.train_pairs(s).theta, ref)


def test_zero_steps_returns_the_start_and_one_step_changes_it():
    s = S.Settings(m=4, pairs=10, steps=0, seed=1)
    start = L.initial_theta(10, 4, s.sigma0, L.Stream(1))
    assert np.array_equal(L.train_pairs(s).theta, start)
    one = L.train_pairs(S.with_(s, steps=1, delta=3.0, sigma0=0.0, pairs=40)).theta
    assert not np.array_equal(one, np.zeros((40, 2, 4)))          # bei Wahrscheinlichkeit 1/4 je Paar trifft mit dem festen Seed mindestens eines (Paare ohne Treffer bleiben bei 0)
    assert (one == 0.0).all(axis=(1, 2)).any()


def test_training_with_a_given_start_does_not_draw_the_start():
    s = S.Settings(m=3, pairs=4, steps=20, seed=2)
    th0 = np.zeros((4, 2, 3))
    st = L.Stream(2)
    L.train_pairs(s, theta0=th0, stream=st)
    assert st.pos == 20 * 2 * 4              # nur die Ziehungen, kein Start


def test_record_returns_the_first_pair_at_the_marked_steps_and_the_end():
    s = S.Settings(m=3, pairs=5, steps=100, seed=4)
    tr = L.train_pairs(s, record_every=30)
    assert tr.trace_steps.tolist() == [0, 30, 60, 90, 100] and tr.trace.shape == (5, 2, 3)
    assert tr.trace[-1] == pytest.approx(PO.softmax(tr.theta)[0], abs=1e-15)
    assert tr.trace.sum(axis=-1) == pytest.approx(np.ones((5, 2)), abs=1e-12)
    assert L.train_pairs(s).trace_steps.size == 0


def test_standard_cell_pairs_settle_on_one_mark():
    """Eigenspiel > 0,99 in der Standardzelle (m = 4, δ = 0, 6 000 Schritte): Mittel über 100 Paare mit großer Marge (gemessen etwa 0,998)."""
    p = PO.softmax(L.train_pairs(S.Settings(pairs=100, seed=12)).theta)
    assert PO.own_play(p[:, 0], p[:, 1]) > 0.99


def test_a_bonus_moves_conventions_to_mark_zero_in_the_distribution():
    """Zweig-Test: δ > 0 ändert den Anteil der Paare auf Marke 0 (Verteilung über 200 Paare, nicht einzelne Läufe)."""
    frac = []
    for delta in (0.0, 3.0):
        p = PO.softmax(L.train_pairs(S.Settings(m=4, delta=delta, pairs=200, steps=3000, seed=21)).theta)
        frac.append(float((np.argmax(p[:, 0] + p[:, 1], axis=1) == 0).mean()))
    assert frac[0] < 0.4 and frac[1] > 0.9


def test_each_other_play_arm_gives_a_different_cross_play_on_at_least_one_cell():
    """Nullspalten-Regel: jeder Arm weicht auf mindestens einer Zelle (m = 4, δ = 0,3) vom Nachbararm ab; Verteilungsaussage über 200 Paare."""
    cross = {}
    for mode in C.MODES:
        p = PO.softmax(L.train_pairs(S.Settings(m=4, delta=0.3, pairs=200, steps=4000, seed=31, mode=mode)).theta)
        cross[mode] = PO.cross_play(p[:, 0], p[:, 1])
    assert abs(cross[C.MODE_SELF] - cross[C.MODE_FIX0]) > 0.1 and abs(cross[C.MODE_SELF] - cross[C.MODE_ALL]) > 0.05


def test_other_play_with_one_fixed_mark_equals_selfplay_for_two_marks_in_distribution():
    """Bei m = 2 ist die Gruppe `Platz 1 fest` trivial, der Arm rechnet also genau das Selbstspiel (gleicher Strom, gleiche Zahlen)."""
    s = S.Settings(m=2, delta=0.3, pairs=30, steps=400, seed=8)
    a = L.train_pairs(s).theta
    b = L.train_pairs(S.with_(s, mode=C.MODE_FIX0)).theta
    assert np.array_equal(a, b)          # keine Schlüssel bei m = 2, der Strom wird gleich aufgeteilt, die Umbenennung ist die Identität


# --- Erwartungs-Update und Fluss ---------------------------------------------------------------------------------------------------------------
def test_expected_update_by_hand(hand_bonus):
    th = np.zeros(2)
    ga, gb = L.expected_update(th, th, hand_bonus["delta"])
    assert ga.tolist() == pytest.approx(list(hand_bonus["update"]), abs=1e-15) and gb.tolist() == pytest.approx(list(hand_bonus["update"]), abs=1e-15)


def test_expected_update_sums_to_zero_and_vanishes_at_a_pure_agreement():
    rng = np.random.default_rng(4)
    ga, gb = L.expected_update(rng.normal(size=5), rng.normal(size=5), 0.7)
    assert ga.sum() == pytest.approx(0.0, abs=1e-15) and gb.sum() == pytest.approx(0.0, abs=1e-15)
    pure = np.array([50.0, 0.0, 0.0])
    ga, gb = L.expected_update(pure, pure, 0.2)
    assert np.abs(ga).max() < 1e-12 and np.abs(gb).max() < 1e-12


def test_flow_with_zero_steps_is_the_start_and_keeps_shape():
    th = np.random.default_rng(0).normal(size=(3, 2, 4))
    assert np.array_equal(L.expected_flow(th, 0.1, 0, 0.1), th)


def test_flow_follows_the_start_favourite_without_bonus():
    """Ohne Bonus gewinnt im deterministischen Fluss die Marke, die beide Spieler am Start zusammen am meisten bevorzugen (Einzelpaare mit großem Abstand)."""
    th = np.zeros((1, 2, 3))
    th[0, :, 1] = 0.3
    out = L.expected_flow(th, 0.0, 3000, 0.1)
    assert PO.softmax(out)[0].argmax(axis=-1).tolist() == [1, 1]
    assert (PO.softmax(out)[0, :, 1] > 0.9).all()


def test_flow_with_a_large_bonus_beats_a_small_start_disadvantage():
    th = np.zeros((1, 2, 3))
    th[0, :, 1] = 0.05
    out = L.expected_flow(th, 3.0, 3000, 0.1)
    assert PO.softmax(out)[0].argmax(axis=-1).tolist() == [0, 0]
