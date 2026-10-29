"""Symmetriegruppen der Markenumbenennung für Other-Play.

Other-Play trainiert gegen einen Partner, dessen Marken je Runde durch eine zufällige Umbenennung φ der Gruppe übersetzt werden: Treffer, wenn a_A = φ(b). Die Gruppe soll genau die
Umbenennungen enthalten, die das Spiel unverändert lassen:
* `alle`: jede der m! Permutationen. Das Spiel ohne Bonus ist unter allen Umbenennungen unverändert (kein Platz ist ausgezeichnet); mit Bonus ist die Gruppe FALSCH.
* `fest0`: Platz 1 (Marke 0) bleibt fest, die übrigen m−1 Marken werden beliebig vertauscht ((m−1)! Permutationen). Das sind die Symmetrien des Spiels MIT Bonus. Bei m = 2 ist die Gruppe trivial.
Die Umbenennung wird aus Gleichverteilten gebildet (argsort, die Gleichstände haben Wahrscheinlichkeit null), damit der Zufall aus SplitMix64 kommt."""

from __future__ import annotations

import math

import numpy as np

import fok_constants as C


def group_size(m: int, group: str) -> int:
    """Zahl der Umbenennungen der Gruppe: m! bzw. (m−1)!."""
    if group == C.GROUP_ALL:
        return math.factorial(m)
    if group == C.GROUP_FIX0:
        return math.factorial(m - 1)
    raise ValueError(f"unbekannte Gruppe: {group}")


def keys_needed(m: int, group: str) -> int:
    """So viele Gleichverteilte braucht eine Umbenennung (je Zeile): m bei `alle`, m−1 bei `fest0` (bei m = 2 keine, die Gruppe ist trivial)."""
    if group == C.GROUP_ALL:
        return m
    if group == C.GROUP_FIX0:
        return m - 1 if m > 2 else 0
    raise ValueError(f"unbekannte Gruppe: {group}")


def random_relabel(keys: np.ndarray, m: int, group: str) -> np.ndarray:
    """Je Zeile eine Umbenennung φ als Feld: φ[z, b] ist die neue Marke von Marke b. `keys` hat Form (Zeilen, keys_needed(m, group)) mit Gleichverteilten.

    `alle`: argsort der Schlüssel liefert eine gleichverteilte Permutation von 0 … m−1. `fest0`: φ[0] = 0, die Marken 1 … m−1 werden aus argsort + 1 gebildet; m = 2: Identität."""
    rows = keys.shape[0]
    if group == C.GROUP_ALL:
        return np.argsort(keys, axis=1)
    if group == C.GROUP_FIX0:
        if m == 2:
            return np.tile(np.arange(2), (rows, 1))
        rest = np.argsort(keys, axis=1) + 1
        return np.concatenate([np.zeros((rows, 1), dtype=np.int64), rest], axis=1)
    raise ValueError(f"unbekannte Gruppe: {group}")


def group_of_mode(mode: str) -> str | None:
    """Symmetriegruppe des Modus: None für das Selbstspiel."""
    if mode == C.MODE_ALL:
        return C.GROUP_ALL
    if mode == C.MODE_FIX0:
        return C.GROUP_FIX0
    if mode == C.MODE_SELF:
        return None
    raise ValueError(f"unbekannter Modus: {mode}")


def other_play_value(p: np.ndarray, q: np.ndarray, delta: float, group: str) -> float:
    """Erwartete Belohnung von Spieler A (Politik p) gegen Spieler B (Politik q) unter zufälliger Umbenennung der Gruppe, mit Bonus δ auf Marke 0:
    `alle`:  1/m + δ·p₀/m (hängt nicht von q ab);
    `fest0`: (1 + δ)·p₀·q₀ + (1 − p₀)(1 − q₀)/(m − 1)."""
    m = len(p)
    if group == C.GROUP_ALL:
        return 1.0 / m + delta * float(p[0]) / m
    if group == C.GROUP_FIX0:
        return (1.0 + delta) * float(p[0]) * float(q[0]) + (1.0 - float(p[0])) * (1.0 - float(q[0])) / (m - 1)
    raise ValueError(f"unbekannte Gruppe: {group}")
