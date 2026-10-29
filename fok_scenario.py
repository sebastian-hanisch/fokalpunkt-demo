"""Szenario: die Einstellungen der App (ein Trainingslauf vieler Paare) und ihre abgeleiteten Größen. Der Zufall kommt aus SplitMix64 (fok_rng)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import fok_constants as C


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Regler der Seitenleiste. Nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld darf nie still Werte verschieben (Lehre aus früheren Demos)."""
    m: int = C.DEFAULT_M                 # Zahl der Übergabeplätze (Marken)
    delta: float = C.DEFAULT_DELTA       # Bonus: Platz 1 bringt 1 + δ statt 1
    eta: float = C.DEFAULT_ETA           # Lernrate
    sigma0: float = C.DEFAULT_SIGMA      # Streuung der Anfangsparameter
    steps: int = C.DEFAULT_STEPS         # Schritte (Runden) des Lernens
    pairs: int = C.DEFAULT_PAIRS         # unabhängig lernende Paare
    t0: int = C.DEFAULT_T0               # Länge der gemeinsamen Vorgeschichte (0: keine)
    mode: str = C.MODE_SELF              # Selbstspiel oder Other-Play mit einer Symmetriegruppe
    seed: int = C.DEFAULT_SEED


def settings_of_preset(name: str) -> Settings:
    return Settings(**C.PRESETS[name])


def with_(s: Settings, **changes) -> Settings:
    """Kopie mit geänderten Feldern (nur Schlüsselwörter)."""
    return replace(s, **changes)


def budget_pairs(s: Settings) -> int:
    """Paare, mit denen die App tatsächlich rechnet: höchstens RUN_BUDGET / Schritte, mindestens 1."""
    return max(1, min(s.pairs, C.RUN_BUDGET // max(1, s.steps)))


def estimated_seconds(s: Settings) -> float:
    """Grobe Zeitschätzung eines Trainingslaufs für die Anzeige am Spinner: Paare · Schritte · (m / 4)^0,5 durch C.PAIR_STEPS_PER_SECOND."""
    return budget_pairs(s) * s.steps * (s.m / 4) ** 0.5 / C.PAIR_STEPS_PER_SECOND
