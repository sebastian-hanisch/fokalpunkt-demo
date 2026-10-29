"""Orakel (ORACLE-PLAYBOOK §1): jede Kernroutine gegen einen ANDEREN Rechenweg, in reinem Python (math, Schleifen), mit scipy (nur hier) oder durch Simulation mit Marge von vier Standardfehlern.

* Fremdspiel-Formel Σ_a p(a)·q(a) gegen Ziehungen (Standardfehler der Trefferquote); Matrixform gegen Doppelschleife.
* Erwartungs-Update: Aufzählung über alle (a, b) gegen die Formel p ⊙ (q ⊙ w − J), gegen zentrale Differenzen der erwarteten Belohnung (mit scipys Softmax) und gegen Simulationsmittel.
* Training (`train_pairs`) gegen eine Schleife in reinem Python mit demselben Gleichverteilten-Strom: Selbstspiel und beide Other-Play-Gruppen; Erwartungsfluss gegen Schleife; Gemeinschaftslernen gegen Schleife.
* Umbenennungen: gültig, Festpunkt, alle Elemente der Gruppe; Other-Play-Erwartung gegen die Aufzählung über die ganze Gruppe (exakt) und gegen Simulation.
* Fremdspiel aus der Verteilung der Konventionen gegen die Aufzählung aller Paarungen.
Alles unter 10 Sekunden. Keine Zeit-Assertion, keine exakten Zufallswerte einzelner Läufe (nur Vergleich zweier Rechenwege auf denselben Zahlen)."""

import itertools
import math

import numpy as np
import pytest

import fok_constants as C
import fok_conventions as CV
import fok_experience as EX
import fok_learning as L
import fok_policy as PO
import fok_scenario as S
import fok_symmetry as SY
from fok_rng import uniform_block

sp = pytest.importorskip("scipy.special")


def py_softmax(theta):
    z = max(theta)
    e = [math.exp(t - z) for t in theta]
    s = sum(e)
    return [x / s for x in e]


def py_draw(p, u):
    cum, total = [], 0.0
    for x in p:
        total += x
        cum.append(total)
    cum = [c / cum[-1] for c in cum]
    return min(sum(1 for c in cum if u >= c), len(p) - 1)


# --- O1 Fremdspiel-Formel -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_match_probability_against_draws(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 9))
    p, q = rng.dirichlet(np.ones(m) * 0.5), rng.dirichlet(np.ones(m) * 0.5)
    n = 200_000
    u = uniform_block(seed + 100, 2 * n)
    a = PO.sample_actions(np.tile(p, (n, 1)), u[:n])
    b = PO.sample_actions(np.tile(q, (n, 1)), u[n:])
    exact = PO.match_probability(p, q)
    se = math.sqrt(max(exact * (1 - exact), 1e-12) / n)
    assert abs((a == b).mean() - exact) < 4 * se


def test_sampling_frequencies_follow_the_policy():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    n = 100_000
    a = PO.sample_actions(np.tile(p, (n, 1)), uniform_block(77, n))
    freq = np.bincount(a, minlength=4) / n
    assert all(abs(f - x) < 4 * math.sqrt(x * (1 - x) / n) for f, x in zip(freq, p))


def test_matrix_form_against_double_loop():
    rng = np.random.default_rng(1)
    pa, pb = PO.softmax(rng.normal(size=(9, 5))), PO.softmax(rng.normal(size=(9, 5)))
    loop = [sum(pa[i, a] * pb[j, a] for a in range(5)) for i in range(9) for j in range(9) if i != j]
    own = [sum(pa[i, a] * pb[i, a] for a in range(5)) for i in range(9)]
    assert PO.cross_play(pa, pb) == pytest.approx(np.mean(loop), rel=1e-12)
    assert PO.own_play(pa, pb) == pytest.approx(np.mean(own), rel=1e-12)


def test_group_cross_play_against_double_loop():
    rng = np.random.default_rng(2)
    pa, pb = PO.softmax(rng.normal(size=(8, 4))), PO.softmax(rng.normal(size=(8, 4)))
    group = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    within = [pa[i] @ pb[j] for i in range(8) for j in range(8) if i != j and group[i] == group[j]]
    between = [pa[i] @ pb[j] for i in range(8) for j in range(8) if group[i] != group[j]]
    w, b = PO.cross_play_groups(pa, pb, group)
    assert w == pytest.approx(np.mean(within), rel=1e-12) and b == pytest.approx(np.mean(between), rel=1e-12)


def test_softmax_against_scipy():
    theta = np.random.default_rng(3).normal(scale=5, size=(6, 7))
    assert PO.softmax(theta) == pytest.approx(sp.softmax(theta, axis=-1), abs=1e-14)


# --- O3 Erwartungs-Update --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_expected_update_by_enumeration_and_central_differences(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 7))
    delta = float(rng.choice([0.0, 0.1, 0.5, 2.0]))
    ta, tb = rng.normal(size=m), rng.normal(size=m)
    pa, pb = py_softmax(ta), py_softmax(tb)
    w = [1.0 + (delta if k == 0 else 0.0) for k in range(m)]
    enum = [0.0] * m                              # Σ_{a,b} p_A(a) p_B(b) r(a,b) (e_a − p_A)
    for a, b in itertools.product(range(m), repeat=2):
        r = w[a] if a == b else 0.0
        for k in range(m):
            enum[k] += pa[a] * pb[b] * r * ((1.0 if k == a else 0.0) - pa[k])
    ga, gb = L.expected_update(ta, tb, delta)
    assert ga.tolist() == pytest.approx(enum, abs=1e-13)

    def j(theta_a, theta_b):
        return float((sp.softmax(theta_a) * sp.softmax(theta_b) * np.array(w)).sum())
    h = 1e-6
    fd_a = [(j(ta + h * np.eye(m)[k], tb) - j(ta - h * np.eye(m)[k], tb)) / (2 * h) for k in range(m)]
    fd_b = [(j(ta, tb + h * np.eye(m)[k]) - j(ta, tb - h * np.eye(m)[k])) / (2 * h) for k in range(m)]
    assert ga.tolist() == pytest.approx(fd_a, abs=1e-8) and gb.tolist() == pytest.approx(fd_b, abs=1e-8)


def test_expected_update_against_the_mean_of_simulated_updates():
    rng = np.random.default_rng(7)
    m, delta, n = 4, 0.5, 400_000
    ta, tb = rng.normal(size=m), rng.normal(size=m)
    pa, pb = PO.softmax(ta), PO.softmax(tb)
    u = uniform_block(5, 2 * n)
    a = PO.sample_actions(np.tile(pa, (n, 1)), u[:n])
    b = PO.sample_actions(np.tile(pb, (n, 1)), u[n:])
    r = L.payoff(a, b, delta)
    upd = r[:, None] * (np.eye(m)[a] - pa)
    ga, _ = L.expected_update(ta, tb, delta)
    se = upd.std(axis=0) / math.sqrt(n)
    assert (np.abs(upd.mean(axis=0) - ga) < 4 * se).all()


# --- O5 Training gegen eine Schleife in reinem Python ---------------------------------------------------------------------------------------
def py_train(s: S.Settings, theta0):
    """Dieselben Gleichverteilten, anderer Rechenweg: Schleifen, math.exp, Ziehung durch Abzählen. Spiegelt die Reihenfolge der Zufallszahlen aus fok_learning."""
    P, m = s.pairs, s.m
    group = SY.group_of_mode(s.mode)
    k = SY.keys_needed(m, group) if group else 0
    per_step = 2 * P + P * k
    theta = [[list(map(float, theta0[i, j])) for j in range(2)] for i in range(P)]
    u_all = uniform_block(s.seed, s.steps * per_step)         # die Anfangsparameter wurden von außen vorgegeben, der Strom beginnt also bei Position 0
    for t in range(s.steps):
        row = u_all[t * per_step: (t + 1) * per_step]
        for i in range(P):
            p = [py_softmax(theta[i][j]) for j in range(2)]
            a = [py_draw(p[j], row[2 * i + j]) for j in range(2)]
            other = a[1]
            if group:
                keys = row[2 * P + i * k: 2 * P + (i + 1) * k]
                if group == C.GROUP_ALL:
                    phi = sorted(range(m), key=lambda x: keys[x])           # argsort
                    # phi[b] ist der b-te Eintrag der Sortierung (np.argsort gibt Indizes in aufsteigender Schlüsselreihenfolge)
                else:
                    phi = [0] + [1 + x for x in sorted(range(m - 1), key=lambda x: keys[x])] if m > 2 else [0, 1]
                other = phi[other]
            r = (1.0 + (s.delta if a[0] == 0 else 0.0)) if a[0] == other else 0.0
            for j in range(2):
                for c in range(m):
                    theta[i][j][c] += s.eta * r * ((1.0 if c == a[j] else 0.0) - p[j][c])
    return np.array(theta)


@pytest.mark.parametrize("mode", list(C.MODES))
@pytest.mark.parametrize("m", [2, 3, 5])
def test_training_against_a_pure_python_loop(mode, m):
    s = S.Settings(m=m, pairs=4, steps=60, delta=0.4, eta=0.2, mode=mode, seed=11 + m)
    theta0 = np.random.default_rng(m).normal(scale=0.5, size=(4, 2, m))
    got = L.train_pairs(s, theta0=theta0).theta
    want = py_train(s, theta0)
    assert got == pytest.approx(want, abs=1e-9)


def test_expected_flow_against_a_python_loop():
    rng = np.random.default_rng(4)
    theta0 = rng.normal(scale=0.4, size=(3, 2, 4))
    delta, eta, steps = 0.3, 0.1, 80
    got = L.expected_flow(theta0, delta, steps, eta)
    for i in range(3):
        ta, tb = list(theta0[i, 0]), list(theta0[i, 1])
        w = [1.0 + delta, 1.0, 1.0, 1.0]
        for _ in range(steps):
            pa, pb = py_softmax(ta), py_softmax(tb)
            jv = sum(pa[k] * pb[k] * w[k] for k in range(4))
            ga = [pa[k] * (pb[k] * w[k] - jv) for k in range(4)]
            gb = [pb[k] * (pa[k] * w[k] - jv) for k in range(4)]
            ta = [ta[k] + eta * ga[k] for k in range(4)]
            tb = [tb[k] + eta * gb[k] for k in range(4)]
        assert got[i, 0].tolist() == pytest.approx(ta, abs=1e-10) and got[i, 1].tolist() == pytest.approx(tb, abs=1e-10)


def test_community_against_a_python_loop():
    m, G, B, rounds, eta, delta, sigma0 = 3, 4, 2, 25, 0.1, 0.5, 0.2
    got = EX.community(m, G, B, rounds, eta, delta, sigma0, L.Stream(21), (rounds,))[rounds]
    u = uniform_block(21, B * G * m * 2 + rounds * 2 * B * G)
    n0 = B * G * m
    z = [math.sqrt(-2.0 * math.log(1.0 - u[i])) * math.cos(2.0 * math.pi * u[n0 + i]) for i in range(n0)]
    theta = [[[sigma0 * z[(b * G + g) * m + c] for c in range(m)] for g in range(G)] for b in range(B)]
    pos = 2 * n0
    w = [1.0 + delta] + [1.0] * (m - 1)
    for _ in range(rounds):
        blk = u[pos: pos + 2 * B * G]
        pos += 2 * B * G
        ua, uk = blk[: B * G], blk[B * G:]
        for b in range(B):
            p = [py_softmax(theta[b][g]) for g in range(G)]
            a = [py_draw(p[g], ua[b * G + g]) for g in range(G)]
            keys = [uk[b * G + g] for g in range(G)]
            perm = sorted(range(G), key=lambda x: keys[x])
            partner = [None] * G
            for x, y in zip(perm[0::2], perm[1::2]):
                partner[x], partner[y] = y, x
            for g in range(G):
                r = w[a[g]] if a[g] == a[partner[g]] else 0.0
                for c in range(m):
                    theta[b][g][c] += eta * r * ((1.0 if c == a[g] else 0.0) - p[g][c])
    pol = np.array([[py_softmax(theta[b][g]) for g in range(G)] for b in range(B)])
    want = EX.community_stats(pol)
    assert got["within"] == pytest.approx(want["within"], abs=1e-9) and got["between"] == pytest.approx(want["between"], abs=1e-9)


# --- O4 Umbenennungen und Other-Play-Erwartung ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("group", [C.GROUP_ALL, C.GROUP_FIX0])
def test_other_play_value_against_the_enumeration_of_the_whole_group(m, group):
    rng = np.random.default_rng(m)
    p, q = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
    delta = 0.7
    elements = [perm for perm in itertools.permutations(range(m)) if group == C.GROUP_ALL or perm[0] == 0]
    assert len(elements) == SY.group_size(m, group)
    total = 0.0
    for phi in elements:
        for a in range(m):
            for b in range(m):
                if a == phi[b]:
                    total += p[a] * q[b] * (1.0 + (delta if a == 0 else 0.0))
    assert SY.other_play_value(p, q, delta, group) == pytest.approx(total / len(elements), abs=1e-13)


def test_other_play_value_against_simulation_with_relabelings_drawn_from_the_stream():
    m, delta, n = 4, 0.5, 300_000
    rng = np.random.default_rng(6)
    p, q = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
    u = uniform_block(31, 2 * n + n * m)
    a = PO.sample_actions(np.tile(p, (n, 1)), u[:n])
    b = PO.sample_actions(np.tile(q, (n, 1)), u[n: 2 * n])
    for group in (C.GROUP_ALL, C.GROUP_FIX0):
        k = SY.keys_needed(m, group)
        phi = SY.random_relabel(u[2 * n: 2 * n + n * k].reshape(n, k), m, group)
        r = L.payoff(a, phi[np.arange(n), b], delta)
        se = r.std() / math.sqrt(n)
        assert abs(r.mean() - SY.other_play_value(p, q, delta, group)) < 4 * se


# --- Konventionen ------------------------------------------------------------------------------------------------------------------------------
def test_cross_from_counts_against_the_enumeration_of_all_pairings():
    labels = [0, 0, 1, 2, 2, 2, 1, 0]
    hits = sum(1 for i, j in itertools.permutations(range(len(labels)), 2) if labels[i] == labels[j])
    counts = CV.convention_counts(np.array(labels), 3)
    assert CV.cross_from_counts(counts) == pytest.approx(hits / (8 * 7), abs=1e-15)


def test_half_way_delta_against_a_dense_interpolation():
    deltas = (0.0, 0.01, 0.1, 1.0, 3.0)
    fr = (0.25, 0.27, 0.40, 0.87, 0.97)
    kind, val = CV.half_way_delta(deltas, fr, 4)
    assert kind == "interpoliert"
    grid = np.exp(np.linspace(math.log(0.1), math.log(1.0), 200_001))
    f = np.interp(np.log(grid), [math.log(0.1), math.log(1.0)], [0.40, 0.87])
    level = (0.25 + 1) / 2
    assert val == pytest.approx(float(grid[np.argmax(f >= level)]), rel=1e-4)
