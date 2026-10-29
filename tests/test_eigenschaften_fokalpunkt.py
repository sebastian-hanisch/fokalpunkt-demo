"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen über das Verfahren, die ohne Kenntnis des richtigen Ergebnisses gelten müssen. Verteilungsaussagen mit Marge, nichts an einzelnen Läufen.

* Eine gemeinsame Verschiebung aller Parameter eines Spielers ändert die Politik nicht (Softmax-Invarianz): gleiche Wahl, gleicher Verlauf (bis auf Rundungsrauschen).
* Umbenennen der Plätze 2 bis m (ohne Bonus alle Plätze) permutiert die gelernte Verteilung der Konventionen: die Verteilung der Konventionen ist symmetrisch (Chi-Quadrat-ähnliche Marge).
* Fremdspiel nach dem Lernen hängt nicht von der Reihenfolge der Paare ab (Permutation der Paare) und ist symmetrisch unter Vertauschen der Rollen A und B.
* Mehr Plätze senken das Fremdspiel ohne Bonus, ein größerer Bonus hebt es (Verteilungsaussagen).
* Gleicher Zufalls-Seed gibt in einem zweiten Prozess (andere Hash-Randomisierung) dieselbe Ausgabe."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fok_constants as C
import fok_conventions as CV
import fok_learning as L
import fok_policy as PO
import fok_scenario as S

ROOT = Path(__file__).resolve().parent.parent


def test_shifting_all_parameters_of_every_player_by_a_constant_gives_the_same_training():
    s = S.Settings(m=4, delta=0.5, pairs=20, steps=300, seed=14)
    th0 = np.random.default_rng(1).normal(scale=0.3, size=(20, 2, 4))
    shift = np.random.default_rng(2).normal(scale=5.0, size=(20, 2, 1))
    a = L.train_pairs(s, theta0=th0).theta
    b = L.train_pairs(s, theta0=th0 + shift).theta
    assert PO.softmax(a) == pytest.approx(PO.softmax(b), abs=1e-9)


def test_without_bonus_the_conventions_are_symmetric_under_renaming_in_distribution():
    """Ohne Bonus ist kein Platz ausgezeichnet: die Konventionen verteilen sich gleichmäßig über die Plätze (500 Paare, m = 4, χ²-ähnliche Marge: jede Zählung innerhalb von vier Standardfehlern von 125)."""
    p = PO.softmax(L.train_pairs(S.Settings(m=4, pairs=500, steps=3000, seed=41)).theta)
    counts = CV.convention_counts(CV.convention_labels(p[:, 0], p[:, 1]), 4)
    se = math.sqrt(500 * 0.25 * 0.75)
    assert all(abs(c - 125) < 4 * se for c in counts)


def test_cross_play_does_not_depend_on_the_order_of_the_pairs():
    p = PO.softmax(L.train_pairs(S.Settings(m=4, pairs=40, steps=500, seed=5)).theta)
    perm = np.random.default_rng(0).permutation(40)
    assert PO.cross_play(p[:, 0], p[:, 1]) == pytest.approx(PO.cross_play(p[perm, 0], p[perm, 1]), abs=1e-12)
    assert PO.own_play(p[:, 0], p[:, 1]) == pytest.approx(PO.own_play(p[perm, 0], p[perm, 1]), abs=1e-12)


def test_cross_play_is_symmetric_when_both_players_swap_roles():
    p = PO.softmax(L.train_pairs(S.Settings(m=3, pairs=30, steps=400, seed=6)).theta)
    assert PO.cross_play(p[:, 0], p[:, 1]) == pytest.approx(PO.cross_play(p[:, 1], p[:, 0]), abs=1e-12)


def test_own_play_exceeds_cross_play_after_learning_in_every_standard_cell():
    """Verteilungsaussage über vier Zellen (m = 2, 4, 8, 16, ohne Bonus, 200 Paare, 3 000 Schritte): das eigene Paar trifft sich um mehr als 0,3 besser als ein Fremder."""
    for m in (2, 4, 8, 16):
        p = PO.softmax(L.train_pairs(S.Settings(m=m, pairs=200, steps=3000, seed=60 + m)).theta)
        assert PO.own_play(p[:, 0], p[:, 1]) - PO.cross_play(p[:, 0], p[:, 1]) > 0.3


def test_more_places_lower_cross_play_without_bonus_in_distribution():
    vals = []
    for m in (2, 4, 8):
        p = PO.softmax(L.train_pairs(S.Settings(m=m, pairs=300, steps=3000, seed=70)).theta)
        vals.append(PO.cross_play(p[:, 0], p[:, 1]))
    assert vals[0] > vals[1] > vals[2]


def test_a_larger_bonus_raises_cross_play_in_distribution():
    vals = []
    for d in (0.0, 0.3, 3.0):
        p = PO.softmax(L.train_pairs(S.Settings(m=4, delta=d, pairs=300, steps=3000, seed=71)).theta)
        vals.append(PO.cross_play(p[:, 0], p[:, 1]))
    assert vals[0] + 0.05 < vals[1] and vals[1] + 0.2 < vals[2]


@pytest.mark.parametrize("hash_seed", ["1", "2"])
def test_same_seed_in_a_second_process_with_other_hash_randomisation_gives_the_same_numbers(hash_seed):
    code = ("import sys; sys.path.insert(0, %r); import fok_learning as L, fok_scenario as S, fok_policy as PO; "
            "th = L.train_pairs(S.Settings(m=3, pairs=5, steps=200, seed=9, mode='Other-Play: Platz 1 fest')).theta; print(repr(float(PO.softmax(th).sum() + th.sum())))" % str(ROOT))
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=120, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    th = L.train_pairs(S.Settings(m=3, pairs=5, steps=200, seed=9, mode=C.MODE_FIX0)).theta
    assert float(out.stdout.strip()) == pytest.approx(float(PO.softmax(th).sum() + th.sum()), abs=1e-9)
