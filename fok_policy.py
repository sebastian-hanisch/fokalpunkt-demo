"""Politiken über m Marken: Softmax, Ziehung aus einer Gleichverteilten, Erfolgswahrscheinlichkeit zweier Politiken (exakt, nicht gezogen).

Eine Politik ist ein Vektor p ∈ ℝ^m mit Summe 1, ein Spieler hat Parameter θ ∈ ℝ^m und p = softmax(θ). Zwei Spieler mit Politiken p und q wählen gleichzeitig eine Marke und
treffen sich mit Wahrscheinlichkeit Σ_a p(a)·q(a). Das ist die einzige Formel, aus der Eigenspiel (derselbe Paarlauf) und Fremdspiel (Spieler aus zwei verschiedenen Paaren) folgen."""

from __future__ import annotations

import numpy as np


def softmax(theta: np.ndarray) -> np.ndarray:
    """Softmax über die letzte Achse, numerisch stabil: das Maximum wird abgezogen, damit exp nie überläuft (große Parameter nach vielen Schritten)."""
    z = theta - theta.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sample_actions(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Zieht je Zeile eine Marke 0 … m−1 mit Wahrscheinlichkeit p aus der Gleichverteilten `u` (Inversion der kumulierten Verteilung: die kleinste Marke mit u < kumuliert, ein u genau auf der Grenze geht zur nächsten Marke, eine Marke mit Wahrscheinlichkeit 0 wird nie gewählt; Form von u = Form von p ohne letzte Achse).

    Die kumulierte Summe wird auf 1 normiert und das Ergebnis auf m−1 begrenzt, damit Rundung in der letzten Stelle nie eine Marke m erzeugt."""
    cum = np.cumsum(p, axis=-1)
    cum = cum / cum[..., -1:]
    a = (u[..., None] >= cum).sum(axis=-1)
    return np.minimum(a, p.shape[-1] - 1)


def match_probability(p: np.ndarray, q: np.ndarray) -> float:
    """Erfolgswahrscheinlichkeit zweier unabhängiger Wahlen aus p und q: Σ_a p(a)·q(a)."""
    return float(np.dot(p, q))


def success_matrix(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Matrix S[i, j] = Σ_a pa_i(a)·pb_j(a): Spieler A des Paares i gegen Spieler B des Paares j. Die Diagonale ist das Eigenspiel, der Rest das Fremdspiel."""
    return pa @ pb.T


def own_play(pa: np.ndarray, pb: np.ndarray) -> float:
    """Eigenspiel: Mittel über die Paare von Σ_a pa_i(a)·pb_i(a) (Spieler A und B desselben Paares)."""
    return float((pa * pb).sum(axis=1).mean())


def cross_play(pa: np.ndarray, pb: np.ndarray) -> float | None:
    """Fremdspiel: Mittel von S[i, j] über alle Paare i ≠ j; bei nur einem Paar gibt es keinen Fremden (None, keine Division durch null)."""
    n = len(pa)
    if n < 2:
        return None
    s = success_matrix(pa, pb)
    return float((s.sum() - np.trace(s)) / (n * (n - 1)))


def cross_play_groups(pa: np.ndarray, pb: np.ndarray, group: np.ndarray) -> tuple[float | None, float | None]:
    """Fremdspiel innerhalb derselben Gruppe (gleiche Vorgeschichte) und zwischen verschiedenen Gruppen: Mittel von S[i, j] über i ≠ j mit group[i] == group[j] bzw. ungleich.
    None, wenn es keine solchen Paarungen gibt."""
    s = success_matrix(pa, pb)
    same = group[:, None] == group[None, :]
    np.fill_diagonal(same, False)
    diff = group[:, None] != group[None, :]
    within = float(s[same].mean()) if same.any() else None
    between = float(s[diff].mean()) if diff.any() else None
    return within, between
