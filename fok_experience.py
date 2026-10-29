"""Geteilte Vorerfahrung: Gruppen von Paaren mit gleicher Vorgeschichte (eine Referenz-Paarung lernt T₀ Schritte, alle Paare der Gruppe starten von ihren Parametern)
und Gemeinschaften (G Spieler lernen zusammen, in jeder Runde bilden sie zufällig Paare).

Vorerfahrung ist hier kein Signal und kein Reden: sie ist ein gemeinsamer Anfangszustand der Lernenden. Die Referenz muss nicht fertig gelernt haben."""

from __future__ import annotations

import numpy as np

import fok_constants as C
import fok_learning as L
import fok_policy as PO
import fok_scenario as S


def group_index(pairs: int, refs: int) -> np.ndarray:
    """Gruppennummer je Paar (zusammenhängende Blöcke, so gleich groß wie möglich): floor(i · refs / pairs)."""
    return (np.arange(pairs) * refs) // pairs


def ref_count(pairs: int) -> int:
    """Zahl der Referenzen der Hauptansicht: höchstens C.REFS, und jede Gruppe hat mindestens zwei Paare (sonst gibt es kein Fremdspiel innerhalb); bei weniger als vier Paaren eine Gruppe."""
    return max(1, min(C.REFS, pairs // 2))


def shared_reference(s: S.Settings, refs: int, per_ref: int, stream: L.Stream, t0: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Anfangsparameter für refs · per_ref Paare mit gemeinsamer Vorgeschichte und die Gruppennummer je Paar.

    Je Referenz lernt eine Paarung `t0` Schritte (Standard: s.t0) aus zufälligem Start (gleicher Modus, gleiche Lernrate und gleicher Bonus wie die Paare); alle per_ref Paare einer Referenz
    starten von deren Parametern. Bei t0 = 0 ist es der gemeinsame zufällige Startpunkt. Rückgabe: (Parameter (refs · per_ref, 2, m), Gruppe (refs · per_ref,))."""
    t0 = s.t0 if t0 is None else t0
    ref = L.train_pairs(s, pairs=refs, steps=t0, stream=stream).theta
    start = np.repeat(ref, per_ref, axis=0)
    group = np.repeat(np.arange(refs), per_ref)
    return start, group


def groups_for_pairs(s: S.Settings, pairs: int, stream: L.Stream) -> tuple[np.ndarray, np.ndarray]:
    """Hauptansicht mit T₀ > 0: `pairs` Paare in ref_count(pairs) zusammenhängenden Gruppen; Anfangsparameter aus der Vorgeschichte der Gruppe, Gruppennummer je Paar."""
    refs = ref_count(pairs)
    group = group_index(pairs, refs)
    ref = L.train_pairs(s, pairs=refs, steps=s.t0, stream=stream).theta
    return ref[group], group


def community(m: int, size: int, communities: int, rounds: int, eta: float, delta: float, sigma0: float, stream: L.Stream, checkpoints: tuple[int, ...]) -> dict:
    """G = `size` Spieler lernen zusammen, in jeder Runde bilden sie zufällig Paare (size gerade); Belohnung 1 bei gleicher Marke (auf Marke 0 1 + δ), Update wie im Paarlernen.

    Rückgabe je Haltepunkt (Runde): {"within": Mittel der Erfolgswahrscheinlichkeit zwischen verschiedenen Spielern DERSELBEN Gemeinschaft, "sure": Anteil der Gemeinschaften mit within > 0,9,
    "between": Mittel zwischen Spielern VERSCHIEDENER Gemeinschaften} (beide exakt aus den Politiken). Zufall je Runde: size · communities Ziehungen, dann size · communities Schlüssel der Paarbildung."""
    if size < 2 or size % 2:
        raise ValueError("die Gemeinschaftsgröße muss gerade und mindestens 2 sein")
    B, G = communities, size
    theta = sigma0 * L.normals(stream, B * G * m).reshape(B, G, m)
    w = L.weights(m, delta)
    eye = np.eye(m)
    bi = np.arange(B)[:, None]
    out: dict[int, dict] = {}
    for t in range(1, max(checkpoints) + 1):
        p = PO.softmax(theta)
        u = stream.take(2 * B * G).reshape(2, B, G)
        a = PO.sample_actions(p, u[0])
        perm = np.argsort(u[1], axis=1)
        partner = np.empty_like(perm)
        left, right = perm[:, 0::2], perm[:, 1::2]
        partner[bi, left] = right
        partner[bi, right] = left
        hit = a == np.take_along_axis(a, partner, axis=1)
        r = hit * w[a]
        theta = theta + eta * r[:, :, None] * (eye[a] - p)
        if t in checkpoints:
            out[t] = community_stats(PO.softmax(theta))
    return out


def community_stats(p: np.ndarray) -> dict:
    """Erfolgswahrscheinlichkeit innerhalb und zwischen Gemeinschaften. p hat Form (Gemeinschaften, G, m)."""
    B, G, m = p.shape
    within = np.empty(B)
    for g in range(B):
        sm = p[g] @ p[g].T
        within[g] = (sm.sum() - np.trace(sm)) / (G * (G - 1))
    flat = p.reshape(B * G, m)
    sm = flat @ flat.T
    grp = np.repeat(np.arange(B), G)
    diff = grp[:, None] != grp[None, :]
    between = float(sm[diff].mean()) if diff.any() else None
    return {"within": float(within.mean()), "sure": float((within > 0.9).mean()), "between": between}
