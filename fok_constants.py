"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): ein Trainingslauf kostet O(Paare · Schritte · m) (je Schritt: Softmax, Ziehung und Update für zwei Spieler je Paar, bei Other-Play
zusätzlich eine Umbenennung je Paar), am Maximum (500 Paare, 6 000 Schritte, m = 16) rund 48 Mio. Spieler-Marken-Operationen je Rechenschritt-Gruppe; die App begrenzt Paare · Schritte
auf RUN_BUDGET und meldet die Kürzung. Die Matrix Spieler A × Spieler B (Fremdspiel exakt) ist O(Paare² · m) und wird einmal je Lauf gebildet (500² · 16 = 4 Mio.).
Erwartungsfluss (E3): O(Schritte · Paare · m). Vorerfahrung: Referenz (T₀ Schritte) plus Weiterlernen aller Paare. Gemeinschaften (E4): O(Runden · Gemeinschaften · G · m)."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
M_MIN, M_MAX, DEFAULT_M = 2, 16, 4              # Zahl der Übergabeplätze (Marken)
DELTA_MIN, DELTA_MAX, DEFAULT_DELTA = 0.0, 3.0, 0.0   # Bonus: Platz 1 spart 1 + δ statt 1
ETA_MIN, ETA_MAX, DEFAULT_ETA = 0.03, 0.3, 0.1  # Lernrate
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.0, 0.5, 0.1   # Streuung der Anfangsparameter
STEPS_MIN, STEPS_MAX, DEFAULT_STEPS = 500, 20_000, 6_000
PAIRS_MIN, PAIRS_MAX, DEFAULT_PAIRS = 1, 500, 300
T0_MIN, T0_MAX, DEFAULT_T0 = 0, 2_000, 0        # Länge der gemeinsamen Vorgeschichte (0: keine, die Paare starten unabhängig)
SEED_MAX, DEFAULT_SEED = 999_999, 35
REFS = 8                                        # Zahl der Referenzen (Gruppen mit gleicher Vorgeschichte) in der Hauptansicht, wenn T₀ > 0
RUN_BUDGET = 3_000_000                          # Paare · Schritte höchstens; darüber rechnet die App mit weniger Paaren und sagt es
PAIR_STEPS_PER_SECOND = 3_000_000               # grobe Schätzung für die Zeitangabe am Spinner (m = 4)
MATRIX_SHOW = 24                                # so viele Paare zeigt die Matrix Spieler A × Spieler B höchstens
TRACE_POINTS = 150                              # Stützstellen des Trainingsverlaufs eines Paares
CHUNK = 100                                     # so viele Schritte werden mit einem Block Zufallszahlen vorab gezogen

# --- Modus ----------------------------------------------------------------------------------------------------------------------------------
MODES = ("Selbstspiel", "Other-Play: alle Plätze", "Other-Play: Platz 1 fest")
MODE_SELF, MODE_ALL, MODE_FIX0 = MODES
GROUP_ALL, GROUP_FIX0 = "alle", "fest0"          # Symmetriegruppen der Umbenennung (fok_symmetry)

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------
EXP_PAIRS = 300                  # Paare je Zelle der Live-Experimente
EXP_SEED = 20261029
DELTAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)     # Gitter von E1 und E2
OP_DELTAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)               # Gitter von E5
MS = (2, 4, 8, 16)
E2_ETAS = (0.03, 0.1, 0.3)
E2_SIGMAS = (0.0, 0.1, 0.5)
E2_STEPS = (2_000, 6_000, 20_000)
E3_DELTAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
E3_MS = (2, 4, 8)
E4_T0S = (0, 25, 50, 100, 200, 400, 1_000)
E4_MS = (2, 4, 8)
E4_REFS, E4_PER_REF = 40, 25                      # vorgerechnet: 40 Referenzen mit je 25 Paaren je Block
E4_LIVE_REFS, E4_LIVE_PER_REF = 8, 20
GROUP_SIZES = (2, 4, 8, 16, 32)                   # Gemeinschaftsgrößen G
GROUP_ROUNDS = (500, 1_000, 2_000)
GROUP_COUNT = 200                                 # Gemeinschaften je Block
SMALL_GROUPS = 100                                # E2: Stichproben aus kleinen Gruppen
SMALL_SETUPS = ((20, 6_000), (40, 20_000))        # (Paare je Stichprobe, Schritte)
SMALL_DELTAS = (0.0, 0.01, 0.05, 0.2)

# --- vorgerechnete Messreihen -------------------------------------------------------------------------------------------------------------------
MEAS_BLOCKS = 4                  # unabhängige Blöcke (Zufallsströme) je Zelle
MEAS_PAIRS = 500                 # Paare je Block

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_OWN = "#2F6B65"            # Eigenspiel
COLOR_CROSS = "#C0762B"          # Fremdspiel
COLOR_LINE = "#4c78a8"
COLOR_RED = "#d62728"
COLOR_GREY = "rgba(150,150,150,0.65)"
MARK_COLORS = ("#C0762B", "#4c78a8", "#2F6B65", "#8c564b", "#9467bd", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2")

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {"m": DEFAULT_M, "delta": DEFAULT_DELTA, "eta": DEFAULT_ETA, "sigma0": DEFAULT_SIGMA, "steps": DEFAULT_STEPS, "pairs": DEFAULT_PAIRS, "t0": DEFAULT_T0,
        "mode": MODE_SELF, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard (m = 4, ohne Bonus)": dict(BASE),
    "Fremde Paare (m = 16)": {**BASE, "m": 16},
    "Kleiner Bonus (δ = 0,1)": {**BASE, "delta": 0.1},
    "Großer Bonus (δ = 1)": {**BASE, "delta": 1.0},
    "Gleiche Vorgeschichte (T₀ = 400)": {**BASE, "t0": 400},
    "Other-Play, Platz 1 fest (m = 8, δ = 0,3)": {**BASE, "m": 8, "delta": 0.3, "mode": MODE_FIX0},
    "Other-Play, falsche Gruppe (m = 4, δ = 0,3)": {**BASE, "delta": 0.3, "mode": MODE_ALL},
}
PRESET_HELP = {
    "Standard (m = 4, ohne Bonus)": "Vier Plätze, kein Platz ausgezeichnet. Jedes Paar legt sich fest (Eigenspiel 0,997), aber ein Fremder trifft nur mit 0,250, so oft wie der Zufall (1/4). Vorgerechnet: 4 Blöcke zu je 500 Paaren.",
    "Fremde Paare (m = 16)": "Sechzehn Plätze: Eigenspiel 0,995, Fremdspiel 0,063 (Zufall: 1/16 = 0,0625). Je mehr Plätze, desto weniger trifft ein Fremder, obwohl jedes Paar sich trifft (4 Blöcke zu je 500 Paaren).",
    "Kleiner Bonus (δ = 0,1)": "Platz 1 spart 10 % mehr Zeit. Bei vier Plätzen hebt das das Fremdspiel nur von 0,250 auf 0,262 (4 Blöcke zu je 500 Paaren); ein kleiner Vorteil ordnet die Paare kaum.",
    "Großer Bonus (δ = 1)": "Platz 1 spart doppelt so viel Zeit. Das Fremdspiel steigt auf 0,750, 86 % der Paare (0,863) legen sich auf Platz 1 fest (vier Plätze, 4 Blöcke zu je 500 Paaren).",
    "Gleiche Vorgeschichte (T₀ = 400)": "Die Paare starten in Gruppen von einer Referenz, die 400 Schritte gelernt hat. Fremde aus derselben Gruppe treffen sich mit 0,994, aus verschiedenen Gruppen mit 0,246; die Referenz selbst hat erst Eigenspiel 0,829 (vier Plätze, E4: 40 Referenzen mit je 25 Paaren, 4 Blöcke). Hier: 8 Referenzen.",
    "Other-Play, Platz 1 fest (m = 8, δ = 0,3)": "Training gegen Partner mit umbenannten Plätzen 2 bis 8, Platz 1 bleibt fest (die Symmetrien des Spiels mit Bonus). Fremdspiel 0,550 statt 0,202 im Selbstspiel; Preis: Eigenspiel 0,767 statt 0,997 (4 Blöcke zu je 500 Paaren).",
    "Other-Play, falsche Gruppe (m = 4, δ = 0,3)": "Alle Plätze werden umbenannt, obwohl Platz 1 ausgezeichnet ist. 95,5 % der Paare (0,955) haben Platz 1 als größte Neigung, aber Eigenspiel und Fremdspiel sind nur 0,251: Treffer nur durch Zufall (1/4). Gegenbeispiel.",
}
