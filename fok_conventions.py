"""Konventionen der Paare und Kennzahlen eines Blocks: auf welche Marke hat sich ein Paar festgelegt, wie verteilen sich die Konventionen, was sagt die Verteilung über das Fremdspiel,
und ab welchem Bonus erreicht der Anteil der Paare auf Marke 0 den halben Weg. Reine Funktionen auf Politiken bzw. Parametern."""

from __future__ import annotations

import math

import numpy as np

import fok_policy as PO


def convention_labels(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Marke, auf die sich ein Paar festgelegt hat: argmax von p_A + p_B. Bei Gleichstand gilt die kleinste Marke (np.argmax nimmt das erste Maximum), das ist festgelegt und getestet."""
    return np.argmax(pa + pb, axis=1)


def convention_counts(labels: np.ndarray, m: int) -> np.ndarray:
    """Zahl der Paare je Marke."""
    return np.bincount(labels, minlength=m)


def cross_from_counts(counts: np.ndarray) -> float | None:
    """Fremdspiel, das die Verteilung der Konventionen bei REINEN Politiken vorhersagt: Σ_a n_a(n_a − 1) / (P(P − 1)); bei einem Paar None. Für große P ist es Σ_a π_a²."""
    p = int(counts.sum())
    if p < 2:
        return None
    return float((counts * (counts - 1)).sum() / (p * (p - 1)))


def cross_null_sd(m: int, pairs: int) -> float | None:
    """Standardabweichung des Fremdspiels um 1/m, wenn die Konventionen der Paare gleichverteilt über die Marken sind (reine Politiken):
    Fremdspiel − 1/m = (X − (m − 1)) / (m (P − 1)) mit X ~ χ²(m − 1), also sd = √(2 (m − 1)) / (m (P − 1)). None bei weniger als zwei Paaren."""
    if pairs < 2:
        return None
    return math.sqrt(2.0 * (m - 1)) / (m * (pairs - 1))


def cross_excess_z(cross: float, m: int, pairs: int) -> float | None:
    """Wie viele Standardabweichungen (der gleichverteilten Konventionen, siehe cross_null_sd) das Fremdspiel über 1/m liegt."""
    sd = cross_null_sd(m, pairs)
    return None if sd is None else (cross - 1.0 / m) / sd


def half_way_delta(deltas, fractions, m: int) -> tuple[str, float | None]:
    """Bonus, bei dem der Anteil der Paare mit Konvention auf Marke 0 den halben Weg von 1/m nach 1 erreicht (Mitte (1/m + 1)/2), log-interpoliert zwischen den Gitterpunkten mit δ > 0.
    Rückgabe (Art, Wert): ("interpoliert", δ), ("unter", erstes δ > 0) wenn schon dort erreicht, ("nie", None) wenn nie."""
    level = (1.0 / m + 1.0) / 2.0
    pos = [(d, f) for d, f in zip(deltas, fractions) if d > 0]
    if pos[0][1] >= level:
        return "unter", float(pos[0][0])
    for (d0, f0), (d1, f1) in zip(pos[:-1], pos[1:]):
        if f0 < level <= f1:
            t = (level - f0) / (f1 - f0)
            return "interpoliert", float(math.exp(math.log(d0) + t * (math.log(d1) - math.log(d0))))
    return "nie", None


def block_metrics(theta: np.ndarray) -> dict:
    """Kennzahlen eines Blocks von Paaren aus den Parametern (Paare, 2, m): Eigenspiel, Fremdspiel (exakt aus den Politiken), Anteil der Paare mit Konvention Marke 0, das aus der Verteilung
    der Konventionen vorhergesagte Fremdspiel, Anteil sicherer Paare (Eigenspiel > 0,9) und mittleres größtes p."""
    p = PO.softmax(theta)
    pa, pb = p[:, 0], p[:, 1]
    m = pa.shape[1]
    labels = convention_labels(pa, pb)
    counts = convention_counts(labels, m)
    own_each = (pa * pb).sum(axis=1)
    return {"own": PO.own_play(pa, pb), "cross": PO.cross_play(pa, pb), "frac0": float((labels == 0).mean()), "pred": cross_from_counts(counts),
            "sure": float((own_each > 0.9).mean()), "pmax": float(((pa + pb) / 2).max(axis=1).mean())}
