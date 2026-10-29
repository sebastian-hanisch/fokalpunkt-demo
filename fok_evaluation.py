"""Kennzahlen der Hauptansicht und Verdikte (reine Funktionen: Art und Text, die App ruft `getattr(st, Art)(Text)`; DEMO-PLAYBOOK §12).

Jedes Verdikt zählt über die Zellen, die es beschreibt (k von m), und nennt die Ausnahme im selben Satz (DEMO-PLAYBOOK §11a). „Fällt“, „bleibt“ und „steigt“ heißen: der Unterschied
zweier Mittel liegt unter −4, innerhalb von ±4 oder über +4 Standardfehlern (Fehler der beiden Mittel quadratisch addiert). Alle Aussagen gelten für dieses Lernmodell (Softmax-REINFORCE),
nicht für Menschen, Sprachmodelle oder echte Übergabeplätze."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import fok_constants as C
import fok_conventions as CV
import fok_experience as EX
import fok_learning as L
import fok_policy as PO
import fok_scenario as S

Z = 4.0                       # Marge in Standardfehlern


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                      # „−0,000“ gibt es nicht
    return text.replace(".", ",").replace("-", "−")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def delta_text(d: float) -> str:
    """Bonus ohne überflüssige Nullen: 0 → „0“, 0,03 → „0,03“, 3,0 → „3“, 0,001 → „0,001“."""
    if d == 0:
        return "0"
    return de(d, 3).rstrip("0").rstrip(",")


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MainView:
    settings: S.Settings
    pairs_used: int                  # Paare, mit denen tatsächlich gerechnet wurde (Budget)
    own: float                       # Eigenspiel
    cross: float | None              # Fremdspiel (None bei einem Paar)
    cross_within: float | None       # Fremdspiel innerhalb einer Gruppe mit gleicher Vorgeschichte (nur bei T₀ > 0)
    cross_between: float | None      # Fremdspiel zwischen Gruppen mit verschiedener Vorgeschichte
    counts: np.ndarray               # (m,) Paare je Konventionsmarke
    pred: float | None               # Fremdspiel aus der Verteilung der Konventionen (reine Politiken)
    sure: float                      # Anteil der Paare mit Eigenspiel > 0,9
    pmax: float                      # mittleres größtes p der beiden Spieler
    matrix: np.ndarray               # (K, K) Erfolgswahrscheinlichkeit Spieler A des Paares i gegen Spieler B des Paares j, K = min(Paare, C.MATRIX_SHOW)
    group: np.ndarray | None         # Gruppe der ersten K Paare (T₀ > 0) oder None
    trace_steps: np.ndarray
    trace: np.ndarray                # (Stützstellen, 2, m) Politiken des ersten Paares

    @property
    def chance(self) -> float:
        return 1.0 / self.settings.m

    @property
    def frac0(self) -> float:
        return float(self.counts[0] / self.counts.sum())


def analyse(s: S.Settings) -> MainView:
    """Ein Trainingslauf mit höchstens S.budget_pairs(s) Paaren (bei T₀ > 0 mit gemeinsamer Vorgeschichte je Gruppe) und die Kennzahlen daraus."""
    used = S.budget_pairs(s)
    su = S.with_(s, pairs=used)
    stream = L.Stream(s.seed)
    group = None
    theta0 = None
    if s.t0 > 0:
        theta0, group = EX.groups_for_pairs(su, used, stream)
    tr = L.train_pairs(su, theta0=theta0, stream=stream, record_every=max(1, s.steps // C.TRACE_POINTS))
    p = PO.softmax(tr.theta)
    pa, pb = p[:, 0], p[:, 1]
    bm = CV.block_metrics(tr.theta)
    counts = CV.convention_counts(CV.convention_labels(pa, pb), s.m)
    within = between = None
    if group is not None:
        within, between = PO.cross_play_groups(pa, pb, group)
    k = min(used, C.MATRIX_SHOW)
    return MainView(settings=s, pairs_used=used, own=bm["own"], cross=bm["cross"], cross_within=within, cross_between=between, counts=counts, pred=bm["pred"], sure=bm["sure"],
                    pmax=bm["pmax"], matrix=PO.success_matrix(pa[:k], pb[:k]), group=None if group is None else group[:k], trace_steps=tr.trace_steps, trace=tr.trace)


# --- Verdikte (reine Funktionen) ------------------------------------------------------------------------------------------------------------------
def _mean(cell: dict, key: str) -> float:
    return cell[key]["mean"]


def verdict_cross(view: MainView) -> tuple[str, str]:
    """Hauptverdikt „Trifft ein Fremder meinen Partner?“: Eigenspiel, Fremdspiel gegen den Zufallswert 1/m und die Wirkung von Bonus, Vorgeschichte und Modus.

    Fälle in dieser Reihenfolge: ein Paar (kein Fremdspiel); gemeinsame Vorgeschichte (Innen- gegen Zwischenspiel); Paare ohne Festlegung (Eigenspiel unter 0,9), mit oder ohne Fremdspiel über dem Zufall;
    Fremdspiel gleich dem Zufall (höchstens vier Standardabweichungen über 1/m); Fremdspiel über dem Zufall."""
    s = view.settings
    chance = view.chance
    if view.cross is None:
        return "info", (f"**Mit einem Paar gibt es keinen Fremden.** Das Fremdspiel ist nicht definiert (es braucht mindestens zwei Paare). Eigenspiel des Paares: {de(view.own, 3)}, "
                        f"Zufallswert 1/m = {de(chance, 3)}.")
    if view.cross_within is not None and view.cross_between is not None and view.own >= 0.9:
        gap = view.cross_within - view.cross_between
        kind = "success" if gap >= 0.2 else "info"
        if gap >= 0.2:
            verdict = "Die Vorgeschichte macht Fremde zu Bekannten."
        elif view.cross_between >= 0.8:
            verdict = "Auch Fremde aus anderen Gruppen treffen sich schon (der Bonus lenkt alle Gruppen auf denselben Platz), die Vorgeschichte ändert daran nichts."
        else:
            verdict = "Der Unterschied ist klein (unter 0,2)."
        return kind, (f"**Gemeinsame Vorgeschichte (T₀ = {s.t0}).** Eigenspiel {de(view.own, 3)}. Fremdspiel innerhalb einer Gruppe mit gleicher Vorgeschichte {de(view.cross_within, 3)}, "
                      f"zwischen Gruppen mit verschiedener Vorgeschichte {de(view.cross_between, 3)}, Zufallswert 1/m = {de(chance, 3)}. {verdict} "
                      f"Mit nur {EX.ref_count(view.pairs_used)} Referenzen streut der Zwischenwert stark.")
    z = CV.cross_excess_z(view.cross, s.m, view.pairs_used)
    sd = CV.cross_null_sd(s.m, view.pairs_used)
    above = z is not None and z > Z
    if view.own < 0.9:
        if above:
            note = (" Das ist der Preis des Other-Play: Paare, die auf einen anderen Platz gelaufen sind, werden durch die Umbenennungen bestraft und bleiben unentschieden."
                    if s.mode == C.MODE_FIX0 else "")
            return "info", (f"**Weniger Festlegung, aber mehr Treffer mit Fremden.** Eigenspiel {de(view.own, 3)} (nur {pct(view.sure, 0)} der Paare treffen sich mit Wahrscheinlichkeit über 0,9), "
                            f"Fremdspiel {de(view.cross, 3)} gegenüber 1/m = {de(chance, 3)}.{note}")
        extra = ""
        if s.mode == C.MODE_ALL and s.delta > 0:
            extra = (" Die Gruppe „alle Plätze“ passt nicht zu einem Spiel mit Bonus: Platz 1 ist ausgezeichnet, die Umbenennung vertauscht ihn trotzdem. "
                     f"Bei {pct(view.frac0, 0)} der Paare ist Platz 1 die größte Neigung, aber ohne Partner, der dieselbe Neigung hat.")
        return "warning", (f"**Die Paare haben sich nicht festgelegt.** Eigenspiel {de(view.own, 3)}, Fremdspiel {de(view.cross, 3)}, Zufallswert 1/m = {de(chance, 3)}; "
                           f"nur {pct(view.sure, 0)} der Paare treffen sich mit Wahrscheinlichkeit über 0,9.{extra}")
    if not above:
        return "info", (f"**Ein Fremder trifft nur zufällig.** Eigenspiel {de(view.own, 3)} (das eigene Paar), Fremdspiel {de(view.cross, 3)}; der Zufallswert ist 1/m = {de(chance, 3)}. "
                        f"Der Abstand liegt innerhalb von {de(Z, 0)} Standardabweichungen (eine Standardabweichung ist {de(sd, 4)} bei {thousands(view.pairs_used)} Paaren, wenn die Paare ihre Plätze gleichverteilt wählen). "
                        f"Jedes Paar hat eine Konvention gelernt, aber nicht dieselbe.")
    pred = de(view.pred, 3) if view.pred is not None else "–"
    return "success", (f"**Ein Fremder trifft häufiger als der Zufall.** Eigenspiel {de(view.own, 3)}, Fremdspiel {de(view.cross, 3)} gegenüber 1/m = {de(chance, 3)}. "
                       f"{pct(view.frac0, 0)} der Paare haben sich auf Platz 1 festgelegt; bei festen Konventionen folgt das Fremdspiel aus ihrer Verteilung ({pred} aus Σ nₐ(nₐ−1)/(P(P−1))).")


def verdict_e1(row: dict, m: int) -> tuple[str, str]:
    """E1 Eigen- gegen Fremdspiel über das Bonus-Gitter von m: row = {δ: Zelle}. Zählt die Zellen mit Eigenspiel ≥ 0,99 und nennt das größte δ, bei dem das Fremdspiel höchstens 0,01 über 1/m liegt."""
    deltas = sorted(row)
    chance = 1.0 / m
    own = [_mean(row[d], "own") for d in deltas]
    cross = [_mean(row[d], "cross") for d in deltas]
    sure = sum(1 for x in own if x >= 0.99)
    near = [d for d, c in zip(deltas, cross) if c - chance <= 0.01]
    over = [d for d, c in zip(deltas, cross) if c - chance > 0.01]
    head = f"**m = {m}:** Eigenspiel ≥ 0,99 in {sure} von {len(deltas)} Zellen (kleinster Wert {de(min(own), 3)}). "
    if not over:
        return "info", head + f"Das Fremdspiel liegt in allen Zellen höchstens 0,01 über 1/m = {de(chance, 3)}."
    first_over = min(over)
    upto = f"bis δ = {delta_text(max(near))}" if near else "in keiner Zelle"
    return "info", head + (f"Fremdspiel höchstens 0,01 über 1/m = {de(chance, 3)} {upto}, ab δ = {delta_text(first_over)} darüber ({de(cross[deltas.index(first_over)], 3)}), "
                           f"bei δ = {delta_text(max(deltas))}: {de(cross[-1], 3)}.")


def verdict_e2_halfway(half: dict[int, tuple[str, float | None]]) -> tuple[str, str]:
    """E2 Halbweg-Bonus nach m: {m: (Art, δ)} aus CV.half_way_delta. Zählt, in wie vielen Schritten von einer Platzzahl zur nächsten der Wert wächst."""
    ms = sorted(half)
    vals = [half[m][1] for m in ms]
    parts = [f"m = {m}: {de(half[m][1], 2)}" if half[m][1] is not None else f"m = {m}: nie" for m in ms]
    steps = [(a, b) for a, b in zip(vals[:-1], vals[1:]) if a is not None and b is not None]
    grow = sum(1 for a, b in steps if b > a)
    return "info", (f"**Halbweg-Bonus** (der Anteil der Paare auf Platz 1 erreicht die Mitte zwischen 1/m und 1): {'; '.join(parts)}. "
                    f"Er wächst in {grow} von {len(steps)} Schritten von einer Platzzahl zur nächsten; es gibt keine scharfe Schwelle, der Anteil steigt glatt mit δ.")


def verdict_e2_sensitivity(half: dict[str, tuple[str, float | None]]) -> tuple[str, str]:
    """E2 Empfindlichkeit bei m = 4: Halbweg-Bonus je Lernrate, Anfangsstreuung und Schrittzahl, {"η = 0,03": (Art, δ), …}. Nennt die Spannweite je Gruppe."""
    out = []
    for prefix, name in (("η", "Lernrate η"), ("σ₀", "Anfangsstreuung σ₀"), ("T", "Schrittzahl")):
        v = [x[1] for k, x in half.items() if k.startswith(prefix) and x[1] is not None]
        if v:
            out.append(f"{name}: {de(min(v), 2)} bis {de(max(v), 2)} (Faktor {de(max(v) / min(v), 1)})")
    return "info", ("**Wovon hängt der Halbweg-Bonus ab?** " + "; ".join(out) + ". Die Lage der Kurve ist eine Eigenschaft des Lernverfahrens (Rauschen gegen Zug), nicht des Spiels; "
                    "drei Werte je Regler ergeben keine Gesetzmäßigkeit.")


def verdict_small_sample(cell: dict) -> tuple[str, str]:
    """E2 Streuung kleiner Stichproben (small_sample): Mittel, Standardabweichung, 5.–95. Perzentil, Größtes, Zahl der Stichproben mit Fremdspiel ≥ 0,81."""
    return "info", (f"**{cell['groups']} Stichproben mit je {cell['pairs_per']} Paaren** (m = {cell['m']}, δ = {delta_text(cell['delta'])}, {thousands(cell['steps'])} Schritte): Fremdspiel im Mittel {de(cell['mean'], 3)}, "
                    f"Standardabweichung {de(cell['sd'], 3)}, 5.–95. Perzentil {de(cell['p5'], 3)} bis {de(cell['p95'], 3)}, größter Wert {de(cell['max'], 3)}; "
                    f"{cell['ge81']} von {cell['groups']} Stichproben liegen bei 0,81 oder darüber. Eine einzelne kleine Stichprobe kann einen Bonus-Effekt vortäuschen.")


def verdict_e3(row: dict, m: int) -> tuple[str, str]:
    """E3 Erwartungsfluss gegen Lauf: row = {δ: Zelle mit det0, sto0, init_det, init_sto}. Zählt die Zellen mit δ > 0, in denen der Fluss mehr als vier Standardfehler (der beiden Mittel) mehr Paare auf
    Platz 1 sieht als der Lauf, und nennt bei δ = 0 den Anteil, mit dem der Lauf dem Start-Favoriten folgt."""
    deltas = sorted(row)
    pos = [d for d in deltas if d > 0]
    gap = {d: _mean(row[d], "det0") - _mean(row[d], "sto0") for d in pos}
    over = [d for d in pos if gap[d] > Z * math.hypot(row[d]["det0"]["se"], row[d]["sto0"]["se"])]
    text = f"**m = {m}:** "
    if 0.0 in row:
        c = row[0.0]
        text += f"Ohne Bonus gewinnt im Fluss der Start-Favorit in {pct(_mean(c, 'init_det'), 1)} der Paare, der Lauf folgt ihm nur in {pct(_mean(c, 'init_sto'), 1)}. "
    text += f"Mit Bonus sieht der Fluss in {len(over)} von {len(pos)} Zellen mehr Paare auf Platz 1 als der Lauf"
    if over:
        d = max(over, key=lambda x: gap[x])
        text += f" (größter Abstand bei δ = {delta_text(d)}: Fluss {pct(_mean(row[d], 'det0'), 1)}, Lauf {pct(_mean(row[d], 'sto0'), 1)})"
    return "info", text + ". Der Zufall des Trainings, nicht der Erwartungswert des Updates, entscheidet, wie groß δ sein muss."


def verdict_e4(row: dict, m: int) -> tuple[str, str]:
    """E4 Vorerfahrung: row = {T₀: Zelle mit within, between}. Nennt das Fremdspiel innerhalb einer Gruppe beim kleinsten und größten T₀, das kleinste T₀ mit Innenspiel ≥ 0,9 und die Spanne des Zwischenspiels."""
    t0s = sorted(row)
    within = [_mean(row[t], "within") for t in t0s]
    between = [_mean(row[t], "between") for t in t0s]
    first90 = next((t for t, w in zip(t0s, within) if w >= 0.9), None)
    text = (f"**m = {m}:** Fremdspiel innerhalb einer Gruppe mit gleicher Vorgeschichte {de(within[0], 3)} (T₀ = {t0s[0]}) bis {de(within[-1], 3)} (T₀ = {t0s[-1]}); "
            f"zwischen Gruppen {de(min(between), 3)} bis {de(max(between), 3)} (Zufallswert 1/m = {de(1.0 / m, 3)}). ")
    text += f"Innenspiel ≥ 0,9 ab T₀ = {first90}." if first90 is not None else "Innenspiel ≥ 0,9 wird im Gitter nicht erreicht."
    return ("success" if first90 is not None else "info"), text


def verdict_e5(arms: dict[str, dict], m: int, delta: float) -> tuple[str, str]:
    """E5 Other-Play: arms = {Arm: {δ: Zelle}} für ein m und ein δ des Gitters. Nennt je Arm Eigen- und Fremdspiel und den Anteil auf Platz 1; die Gruppe „alle Plätze“ mit Bonus ist das Gegenbeispiel."""
    parts = []
    for name, row in arms.items():
        c = row[delta]
        parts.append(f"{name}: Eigenspiel {de(_mean(c, 'own'), 3)}, Fremdspiel {de(_mean(c, 'cross'), 3)}, Konvention Platz 1 bei {pct(_mean(c, 'frac0'), 0)}")
    text = f"**m = {m}, δ = {delta_text(delta)}.** " + "; ".join(parts) + ". "
    if "alle Plätze" in arms and delta > 0:
        c = arms["alle Plätze"][delta]
        if _mean(c, "own") < 0.5 and _mean(c, "frac0") > 0.5:
            text += "Mit der falschen Gruppe „alle Plätze“ landen die meisten Paare bei Platz 1 und trotzdem nicht beieinander: Other-Play braucht die richtige Gruppe."
    return "info", text


def float_keys(d: dict) -> dict:
    """Wörterbuch aus der JSON-Datei (Textschlüssel wie "0.01") mit Zahlenschlüsseln (0.01), Reihenfolge wie im Original."""
    return {float(k): v for k, v in d.items()}


def int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}
