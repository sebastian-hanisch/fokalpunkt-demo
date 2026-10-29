"""Lernen im Zuordnungsspiel: Softmax-REINFORCE ohne Basis, vektorisiert über viele unabhängige Paare, dazu Erwartungs-Update und Erwartungsfluss.

Zwei Spieler wählen gleichzeitig eine von m Marken (Übergabeplätzen). Gleiche Marke: Belohnung 1, auf Marke 0 (Platz 1) 1 + δ; sonst 0. Jeder Spieler hat Parameter θ ∈ ℝ^m und die Politik
p = softmax(θ). Update nach jeder Runde (Williams 1992, ohne Basis): θ ← θ + η·r·(e_a − p), mit der gewählten Marke a und der Belohnung r (nur der Spieler selbst sieht seine Marke und r).

**Reihenfolge der Zufallszahlen** (SplitMix64, `Stream`): erst die Anfangsparameter (Box-Muller aus 2 · Paare · 2 · m Gleichverteilten: erst alle u₁, dann alle u₂), dann je Schritt
2 · Paare Gleichverteilte für die Ziehungen (Form (Paare, 2): Spieler A, Spieler B) und bei Other-Play Paare · k Schlüssel für die Umbenennung (k = fok_symmetry.keys_needed).
Die Schritte werden in Blöcken von C.CHUNK vorab gezogen; die Blockgröße ändert die Zahlenfolge nicht (`Stream.take(a)` und `take(b)` zusammen sind `take(a + b)`).

**Other-Play** (Hu u. a., „Other-Play“ for Zero-Shot Coordination, im Lehrbuchmodell): die Marke von Spieler B wird je Runde durch eine zufällige Umbenennung φ der Symmetriegruppe übersetzt,
Treffer bei a_A = φ(a_B). Beide Spieler lernen mit der Belohnung r und ihrer eigenen (nicht übersetzten) Marke."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fok_constants as C
import fok_policy as PO
import fok_scenario as S
import fok_symmetry as SY
from fok_rng import _GAMMA, _MASK, uniform_block


# --------------------------------------------------------------------------------- Zufall
def block_seed(seed: int, offset: int) -> int:
    """Seed des SplitMix64-Stroms ab Position `offset` (der Generator ist ein Zähler: Seed + offset·γ)."""
    return (seed + offset * _GAMMA) & _MASK


class Stream:
    """Fortlaufender SplitMix64-Strom in Blöcken: `take(count)` liefert die nächsten `count` Gleichverteilten, `Stream(seed).take(a)` und danach `take(b)` zusammen genau `uniform_block(seed, a + b)`."""

    def __init__(self, seed: int):
        self.seed = seed & _MASK
        self.pos = 0

    def take(self, count: int) -> np.ndarray:
        u = uniform_block(block_seed(self.seed, self.pos), count)
        self.pos += count
        return u


def normals(stream: Stream, count: int) -> np.ndarray:
    """Standardnormalverteilte (Box-Muller): z = √(−2 ln(1 − u₁))·cos(2π u₂) aus zwei Blöcken Gleichverteilter; 1 − u₁ liegt in (0, 1], der Logarithmus ist endlich."""
    u1 = stream.take(count)
    u2 = stream.take(count)
    return np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * np.pi * u2)


# --------------------------------------------------------------------------------- Einheiten
def initial_theta(pairs: int, m: int, sigma0: float, stream: Stream) -> np.ndarray:
    """Anfangsparameter aller Spieler, Form (Paare, 2, m): Normalverteilte mit Streuung σ₀ (bei σ₀ = 0 alles 0, also gleichverteilte Politiken). Es werden immer Zufallszahlen verbraucht."""
    return sigma0 * normals(stream, pairs * 2 * m).reshape(pairs, 2, m)


def payoff(a_own: np.ndarray, a_other: np.ndarray, delta: float) -> np.ndarray:
    """Belohnung des Spielers mit Marke `a_own` gegen die (übersetzte) Marke `a_other`: 1 bei Treffer, auf Marke 0 1 + δ, sonst 0."""
    hit = a_own == a_other
    return hit * (1.0 + delta * (a_own == 0))


def reinforce_step(theta: np.ndarray, p: np.ndarray, a: np.ndarray, reward: np.ndarray, eta: float) -> np.ndarray:
    """Ein REINFORCE-Schritt ohne Basis für alle Paare und beide Spieler: θ + η·r·(e_a − p). Form von θ und p (Paare, 2, m), von a (Paare, 2), von r (Paare,)."""
    onehot = np.eye(theta.shape[-1])[a]
    return theta + eta * reward[:, None, None] * (onehot - p)


@dataclass(frozen=True)
class Trained:
    theta: np.ndarray                  # (Paare, 2, m) Parameter am Ende
    trace_steps: np.ndarray            # (K,) Schrittzahlen der Stützstellen (0 = Anfang); leer ohne Aufzeichnung
    trace: np.ndarray                  # (K, 2, m) Politiken des ersten Paares an den Stützstellen


def train_pairs(s: S.Settings, *, theta0: np.ndarray | None = None, stream: Stream | None = None, pairs: int | None = None, steps: int | None = None,
                record_every: int = 0) -> Trained:
    """Trainiert `pairs` unabhängige Paare (Standard: s.pairs) `steps` Schritte (Standard: s.steps) gegeneinander.

    theta0: Anfangsparameter (sonst aus `stream` gezogen); stream: Zufallsstrom (sonst Stream(s.seed)); record_every > 0 zeichnet die Politiken des ersten Paares alle so viele Schritte auf (und am Ende)."""
    P = s.pairs if pairs is None else pairs
    T = s.steps if steps is None else steps
    m = s.m
    stream = Stream(s.seed) if stream is None else stream
    theta = initial_theta(P, m, s.sigma0, stream) if theta0 is None else np.array(theta0, dtype=float)
    group = SY.group_of_mode(s.mode)
    k = SY.keys_needed(m, group) if group else 0
    per_step = 2 * P + P * k
    rows = np.arange(P)
    steps_rec: list[int] = []
    rec: list[np.ndarray] = []

    def record(step: int, p: np.ndarray):
        steps_rec.append(step)
        rec.append(p[0].copy())

    if record_every:
        record(0, PO.softmax(theta))
    done = 0
    while done < T:
        n = min(C.CHUNK, T - done)
        u = stream.take(n * per_step).reshape(n, per_step)
        for i in range(n):
            p = PO.softmax(theta)
            a = PO.sample_actions(p, u[i, : 2 * P].reshape(P, 2))
            other = a[:, 1]
            if group:
                phi = SY.random_relabel(u[i, 2 * P:].reshape(P, k), m, group)
                other = phi[rows, other]
            theta = reinforce_step(theta, p, a, payoff(a[:, 0], other, s.delta), s.eta)
            done += 1
            if record_every and done % record_every == 0 and done != T:
                record(done, PO.softmax(theta))
    if record_every:
        record(T, PO.softmax(theta))
    return Trained(theta=theta, trace_steps=np.array(steps_rec, dtype=np.int64), trace=np.array(rec) if rec else np.zeros((0, 2, m)))


# --------------------------------------------------------------------------------- Erwartungswert des Updates
def weights(m: int, delta: float) -> np.ndarray:
    """Belohnung bei Treffer je Marke: 1, auf Marke 0 1 + δ."""
    w = np.ones(m)
    w[0] += delta
    return w


def expected_update(theta_a: np.ndarray, theta_b: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Erwartungswert des REINFORCE-Updates beider Spieler (Selbstspiel) in Parametern: für A, mit J = Σ_a p_A(a)·p_B(a)·w(a), ist E[Δθ_A]/η = p_A ⊙ (p_B ⊙ w − J);
    für B ebenso mit vertauschten Rollen. Das ist der Gradient der erwarteten Belohnung J nach θ (Softmax)."""
    pa, pb = PO.softmax(theta_a), PO.softmax(theta_b)
    w = weights(len(pa), delta)
    j = float((pa * pb * w).sum())
    return pa * (pb * w - j), pb * (pa * w - j)


def expected_flow(theta0: np.ndarray, delta: float, steps: int, eta: float) -> np.ndarray:
    """Deterministischer Erwartungsfluss: `steps` Schritte θ ← θ + η·E[Δθ]/η (Euler) für alle Paare zugleich, Form (Paare, 2, m). Kein Zufall, also vom Start allein bestimmt."""
    theta = np.array(theta0, dtype=float)
    w = weights(theta.shape[-1], delta)
    for _ in range(steps):
        p = PO.softmax(theta)
        j = (p[:, 0, :] * p[:, 1, :] * w).sum(axis=1, keepdims=True)
        ga = p[:, 0, :] * (p[:, 1, :] * w - j)
        gb = p[:, 1, :] * (p[:, 0, :] * w - j)
        theta[:, 0, :] += eta * ga
        theta[:, 1, :] += eta * gb
    return theta
