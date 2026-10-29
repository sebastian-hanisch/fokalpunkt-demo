"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_match():
    """Erfolgswahrscheinlichkeit Σ_a p(a)·q(a) von Hand:
    p = (0,5; 0,5), q = (1; 0): 0,5·1 + 0,5·0 = 0,5.
    p = (0,2; 0,8), q = (0,5; 0,5): 0,2·0,5 + 0,8·0,5 = 0,1 + 0,4 = 0,5.
    drei Marken, p gleichverteilt, q = (0,5; 0,5; 0): (1/3)·0,5 + (1/3)·0,5 + 0 = 1/3.
    p = q = (0,5; 0,5) mit Bonus δ = 0,2 (Platz 1 zählt 1,2): 0,25·1,2 + 0,25·1 = 0,3 + 0,25 = 0,55."""
    return [
        ((0.5, 0.5), (1.0, 0.0), 0.5),
        ((0.2, 0.8), (0.5, 0.5), 0.5),
        ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0.0), 1 / 3),
    ]


@pytest.fixture
def hand_bonus():
    """p = q = (0,5; 0,5), δ = 0,2: J = 0,25·1,2 + 0,25·1 = 0,55. Erwartungs-Update von Spieler A in Parametern: p ⊙ (q ⊙ w − J) = 0,5·(0,6 − 0,55), 0,5·(0,5 − 0,55) = (0,025; −0,025)."""
    return {"delta": 0.2, "value": 0.55, "update": (0.025, -0.025)}


@pytest.fixture
def hand_two_pairs():
    """Zwei reine Paare mit verschiedener Konvention: Paar 0 beide auf Marke 0, Paar 1 beide auf Marke 1 (m = 2).
    Eigenspiel: jedes Paar trifft sich mit Wahrscheinlichkeit 1, Mittel 1. Fremdspiel: A von Paar 0 gegen B von Paar 1 und umgekehrt: 0, Mittel 0."""
    pa = np.array([[1.0, 0.0], [0.0, 1.0]])
    return {"pa": pa, "pb": pa.copy(), "own": 1.0, "cross": 0.0}


@pytest.fixture
def hand_counts():
    """Fünf reine Paare, drei mit Konvention Marke 0, zwei mit Marke 1 (m = 2): Fremdspiel = Σ n_a (n_a − 1) / (P (P − 1)) = (3·2 + 2·1) / (5·4) = 8/20 = 0,4.
    Gegenprobe: unter den 20 geordneten Paaren (i, j), i ≠ j, haben 6 + 2 = 8 dieselbe Konvention."""
    return {"labels": np.array([0, 0, 0, 1, 1]), "m": 2, "cross": 0.4}


@pytest.fixture
def hand_step():
    """Ein REINFORCE-Schritt von Hand: m = 2, θ = (0, 0), p = (0,5; 0,5), Marke a = 0, Belohnung r = 1,2 (Treffer auf Marke 0 mit δ = 0,2), η = 0,1:
    Δθ = 0,1·1,2·(e₀ − p) = 0,12·(0,5; −0,5) = (0,06; −0,06)."""
    return {"theta": np.zeros((1, 2, 2)), "a": np.array([[0, 0]]), "reward": np.array([1.2]), "eta": 0.1, "delta": (0.06, -0.06)}


@pytest.fixture
def hand_softmax():
    """Softmax von Hand: θ = (0, ln 2) → p = (1/3; 2/3). θ = (0, 0, 0) → (1/3, 1/3, 1/3). Verschiebung aller Parameter ändert nichts: (1000, 1000 + ln 2) → (1/3; 2/3)."""
    return {"ln2": np.log(2.0)}


@pytest.fixture
def hand_relabel():
    """m = 3, Gruppe `alle` mit Schlüsseln (0,7; 0,1; 0,4): argsort = (1, 2, 0), das ist die Permutation 'Marke 0 → 1, 1 → 2, 2 → 0'.
    Gruppe `fest0` mit m = 3 und Schlüsseln (0,9; 0,2): argsort = (1, 0), plus 1 = (2, 1), vorn die 0: φ = (0, 2, 1) (Marke 0 fest, 1 und 2 vertauscht). Bei m = 2 ist φ die Identität (0, 1)."""
    return {"keys_all": np.array([[0.7, 0.1, 0.4]]), "all": [1, 2, 0], "keys_fix": np.array([[0.9, 0.2]]), "fix": [0, 2, 1]}


@pytest.fixture
def hand_halfway():
    """m = 2, Gitter δ = (0; 0,1; 1) und Anteile (0,5; 0,6; 0,9): Halbweg-Niveau (1/2 + 1)/2 = 0,75. Zwischen δ = 0,1 (0,6) und δ = 1 (0,9) log-interpoliert:
    t = (0,75 − 0,6)/(0,9 − 0,6) = 0,5, δ = exp(ln 0,1 + 0,5·(ln 1 − ln 0,1)) = 0,1·10^0,5 = 0,316228."""
    return {"deltas": (0.0, 0.1, 1.0), "fractions": (0.5, 0.6, 0.9), "m": 2, "value": 0.1 * 10 ** 0.5}
