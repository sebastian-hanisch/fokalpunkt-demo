"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12, Punkt 2): Art, jede Zahl im Text und jeder Zweig an handgesetzten Werten; Formate."""

import numpy as np

import fok_constants as C
import fok_evaluation as V
import fok_scenario as S


def mv(m=4, *, own=0.997, cross=0.25, within=None, between=None, pairs=300, frac0=0.25, sure=1.0, pred=0.25, mode=C.MODE_SELF, delta=0.0, t0=0):
    s = S.Settings(m=m, pairs=pairs, mode=mode, delta=delta, t0=t0)
    counts = np.round(np.array([frac0] + [(1 - frac0) / (m - 1)] * (m - 1)) * pairs).astype(int)
    return V.MainView(settings=s, pairs_used=pairs, own=own, cross=cross, cross_within=within, cross_between=between, counts=counts, pred=pred, sure=sure, pmax=0.99,
                      matrix=np.eye(2), group=None, trace_steps=np.array([0, 1]), trace=np.zeros((2, 2, m)))


def cell(**kw):
    return {k: {"mean": v, "se": 0.001} for k, v in kw.items()}


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def test_de_thousands_pct():
    assert V.de(1.5, 1) == "1,5" and V.de(3.14159, 3) == "3,142" and V.de(7, 0) == "7"
    assert V.de(-0.91, 3) == "−0,910" and V.de(-0.0001, 3) == "0,000"
    assert V.thousands(1234567) == "1.234.567" and V.thousands(999.6) == "1.000" and V.thousands(0) == "0"
    assert V.pct(0.0943) == "9,4 %" and V.pct(0.0, 0) == "0 %" and V.pct(1.0) == "100,0 %"


def test_key_conversions_keep_the_order():
    assert list(V.float_keys({"0.0": 1, "0.001": 2, "3.0": 3})) == [0.0, 0.001, 3.0]
    assert list(V.int_keys({"0": 1, "400": 2, "25": 3})) == [0, 400, 25]


# --- Hauptverdikt ----------------------------------------------------------------------------------------------------------------------------
def test_cross_one_pair_has_no_cross_play():
    kind, text = V.verdict_cross(mv(cross=None, pairs=1, pred=None))
    assert kind == "info" and "Mit einem Paar gibt es keinen Fremden" in text and "nicht definiert" in text and "0,997" in text and "0,250" in text


def test_cross_at_chance():
    kind, text = V.verdict_cross(mv(cross=0.251))
    assert kind == "info" and text.startswith("**Ein Fremder trifft nur zufällig.**") and "Fremdspiel 0,251" in text and "1/m = 0,250" in text and "0,0020" in text and "300 Paaren" in text


def test_cross_boundary_is_four_standard_deviations_inclusive():
    sd = (2 * 3) ** 0.5 / (4 * 299)
    assert V.verdict_cross(mv(cross=0.25 + 4 * sd - 1e-9))[0] == "info"
    assert V.verdict_cross(mv(cross=0.25 + 4 * sd + 1e-4))[0] == "success"


def test_cross_above_chance():
    kind, text = V.verdict_cross(mv(cross=0.75, frac0=0.86, pred=0.749, delta=1.0))
    assert kind == "success" and "häufiger als der Zufall" in text and "Fremdspiel 0,750 gegenüber 1/m = 0,250" in text and "86 % der Paare" in text and "0,749" in text


def test_cross_paths_without_pairs_that_settled():
    kind, text = V.verdict_cross(mv(own=0.25, cross=0.25, sure=0.07, frac0=0.96, mode=C.MODE_ALL, delta=0.3))
    assert kind == "warning" and "nicht festgelegt" in text and "nur 7 % der Paare" in text and "alle Plätze" in text and "96 %" in text
    kind, text = V.verdict_cross(mv(own=0.25, cross=0.25, sure=0.07, mode=C.MODE_SELF))
    assert kind == "warning" and "alle Plätze" not in text


def test_cross_with_less_commitment_but_more_hits_than_chance():
    kind, text = V.verdict_cross(mv(m=8, own=0.775, cross=0.559, sure=0.74, mode=C.MODE_FIX0, delta=0.3, pred=0.5))
    assert kind == "info" and "Weniger Festlegung" in text and "Eigenspiel 0,775" in text and "0,559" in text and "Preis des Other-Play" in text
    assert "Preis des Other-Play" not in V.verdict_cross(mv(m=8, own=0.775, cross=0.559, sure=0.74, mode=C.MODE_SELF))[1]


def test_cross_with_common_history():
    kind, text = V.verdict_cross(mv(within=0.998, between=0.322, delta=0.0, t0=400))
    assert kind == "success" and "T₀ = 400" in text and "0,998" in text and "0,322" in text and "Fremde zu Bekannten" in text and "8 Referenzen" in text
    kind, text = V.verdict_cross(mv(within=0.4, between=0.3, t0=25))
    assert kind == "info" and "unter 0,2" in text
    kind, text = V.verdict_cross(mv(within=0.999, between=0.998, delta=1.0, t0=400))
    assert kind == "info" and "der Bonus lenkt alle Gruppen auf denselben Platz" in text and "unter 0,2" not in text


def test_cross_with_history_boundary_gap_is_inclusive():
    assert V.verdict_cross(mv(within=0.5, between=0.3, t0=25))[0] == "success"
    assert V.verdict_cross(mv(within=0.4999, between=0.3, t0=25))[0] == "info"


# --- E1 ----------------------------------------------------------------------------------------------------------------------------------------
def e1_row(cross_values, own=0.997):
    return {d: cell(own=own, cross=c) for d, c in zip(C.DELTAS, cross_values)}


def test_e1_names_the_last_bonus_near_chance_and_the_first_above():
    row = e1_row([0.250, 0.249, 0.251, 0.250, 0.251, 0.262, 0.361, 0.750, 0.949])
    kind, text = V.verdict_e1(row, 4)
    assert kind == "info" and "9 von 9 Zellen" in text and "kleinster Wert 0,997" in text and "bis δ = 0,03," in text and "ab δ = 0,1 darüber (0,262)" in text and "bei δ = 3: 0,949" in text


def test_e1_without_any_cell_above_chance():
    kind, text = V.verdict_e1(e1_row([0.25] * 9), 4)
    assert "in allen Zellen höchstens 0,01 über 1/m = 0,250" in text


def test_e1_counts_cells_with_low_own_play():
    row = e1_row([0.25] * 9)
    row[0.3] = cell(own=0.9, cross=0.25)
    assert "8 von 9 Zellen (kleinster Wert 0,900)" in V.verdict_e1(row, 4)[1]


def test_e1_when_every_cell_is_above_chance():
    assert "in keiner Zelle" in V.verdict_e1(e1_row([0.5] * 9), 4)[1]


# --- E2 ----------------------------------------------------------------------------------------------------------------------------------------
def test_e2_halfway_counts_the_growth():
    kind, text = V.verdict_e2_halfway({2: ("interpoliert", 0.25), 4: ("interpoliert", 0.41), 8: ("interpoliert", 0.55), 16: ("interpoliert", 0.73)})
    assert kind == "info" and "m = 2: 0,25; m = 4: 0,41; m = 8: 0,55; m = 16: 0,73" in text and "3 von 3 Schritten" in text
    text = V.verdict_e2_halfway({2: ("interpoliert", 0.5), 4: ("interpoliert", 0.4), 8: ("nie", None)})[1]
    assert "1 von 1 Schritten" not in text and "0 von 1 Schritten" in text and "m = 8: nie" in text


def test_e2_sensitivity_gives_the_span_per_group():
    half = {"η = 0,03": ("interpoliert", 0.21), "η = 0,3": ("interpoliert", 1.14), "σ₀ = 0": ("interpoliert", 0.37), "σ₀ = 0,5": ("interpoliert", 0.98),
            "T = 2.000": ("interpoliert", 0.43), "T = 20.000": ("interpoliert", 0.41)}
    kind, text = V.verdict_e2_sensitivity(half)
    assert kind == "info" and "Lernrate η: 0,21 bis 1,14 (Faktor 5,4)" in text and "Anfangsstreuung σ₀: 0,37 bis 0,98 (Faktor 2,6)" in text and "Schrittzahl: 0,41 bis 0,43 (Faktor 1,0)" in text


def test_small_sample_verdict_quotes_the_cell():
    c = {"groups": 100, "pairs_per": 20, "m": 2, "delta": 0.2, "steps": 6000, "mean": 0.576, "sd": 0.095, "p5": 0.479, "p95": 0.731, "max": 0.899, "ge81": 1}
    kind, text = V.verdict_small_sample(c)
    assert kind == "info" and "100 Stichproben mit je 20 Paaren" in text and "m = 2, δ = 0,2, 6.000 Schritte" in text and "im Mittel 0,576" in text and "0,479 bis 0,731" in text and "größter Wert 0,899" in text and "1 von 100" in text


# --- E3 ----------------------------------------------------------------------------------------------------------------------------------------
def e3_row(det, sto):
    out = {}
    for d, a, b in zip(C.E3_DELTAS, det, sto):
        out[d] = {"det0": {"mean": a, "se": 0.005}, "sto0": {"mean": b, "se": 0.005}, "init_det": {"mean": 0.97, "se": 0.0}, "init_sto": {"mean": 0.39, "se": 0.0}}
    return out


def test_e3_counts_cells_where_the_flow_overestimates():
    row = e3_row([0.25, 0.28, 0.34, 0.63, 0.975, 1.0], [0.246, 0.253, 0.279, 0.356, 0.521, 0.863])
    kind, text = V.verdict_e3(row, 4)
    assert kind == "info" and "gewinnt im Fluss der Start-Favorit in 97,0 %" in text and "nur in 39,0 %" in text and "in 4 von 5 Zellen" in text
    assert "größter Abstand bei δ = 0,3: Fluss 97,5 %, Lauf 52,1 %" in text


def test_e3_when_the_flow_never_overestimates():
    text = V.verdict_e3(e3_row([0.25] * 6, [0.25] * 6), 4)[1]
    assert "in 0 von 5 Zellen" in text and "größter Abstand" not in text


# --- E4 ----------------------------------------------------------------------------------------------------------------------------------------
def test_e4_reports_the_first_history_with_inside_play_above_0_9():
    row = {t: cell(within=w, between=b) for t, w, b in zip(C.E4_T0S, (0.286, 0.431, 0.494, 0.712, 0.876, 0.994, 0.998), (0.249, 0.249, 0.251, 0.248, 0.257, 0.246, 0.24))}
    kind, text = V.verdict_e4(row, 4)
    assert kind == "success" and "0,286 (T₀ = 0) bis 0,998 (T₀ = 1000)" in text and "zwischen Gruppen 0,240 bis 0,257" in text and "ab T₀ = 400" in text


def test_e4_without_reaching_0_9():
    row = {t: cell(within=0.3, between=0.25) for t in (0, 25)}
    kind, text = V.verdict_e4(row, 4)
    assert kind == "info" and "wird im Gitter nicht erreicht" in text


# --- E5 ----------------------------------------------------------------------------------------------------------------------------------------
def test_e5_names_the_counterexample_only_for_the_wrong_group_with_a_bonus():
    arms = {"Selbstspiel": {0.3: cell(own=0.997, cross=0.202, frac0=0.385)}, "alle Plätze": {0.3: cell(own=0.127, cross=0.127, frac0=0.864)}, "Platz 1 fest": {0.3: cell(own=0.767, cross=0.550, frac0=0.734)}}
    kind, text = V.verdict_e5(arms, 8, 0.3)
    assert kind == "info" and "Selbstspiel: Eigenspiel 0,997, Fremdspiel 0,202, Konvention Platz 1 bei 38 %" in text and "Platz 1 fest: Eigenspiel 0,767, Fremdspiel 0,550" in text
    assert "braucht die richtige Gruppe" in text
    arms0 = {k: {0.0: v[0.3]} for k, v in arms.items()}
    assert "braucht die richtige Gruppe" not in V.verdict_e5(arms0, 8, 0.0)[1]
    arms["alle Plätze"][0.3] = cell(own=0.9, cross=0.9, frac0=0.9)
    assert "braucht die richtige Gruppe" not in V.verdict_e5(arms, 8, 0.3)[1]
