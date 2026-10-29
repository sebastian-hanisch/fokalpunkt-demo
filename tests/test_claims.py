"""Jede Zahl aus der README und den Hilfetexten, nachgerechnet (DEMO-PLAYBOOK §4).

Zwei Quellen, beide plattformsicher:
* die gespeicherten Messreihen (`data/measurements.json`, 4 Blöcke zu je 500 Paaren je Zelle, Erzeuger-Hash geprüft in `test_measurements.py`): die README-Zahl ist der gespeicherte Wert, gerundet;
  eine Stichprobe der Zellen wird in `test_measurements.py` mit anderen Seeds innerhalb einer Marge reproduziert. Das Lesen der Datei hängt nicht von der Plattform ab.
* exakte Rechnung (Formeln, Hand-Instanzen): Gleitkommagleichheit nur mit `approx`.
Keine Zeit-Assertion, keine exakten Zufallswerte einzelner Läufe, Argmax nur dort, wo der Abstand weit über dem Standardfehler liegt. Die Rangfolgen und Zählungen über das Raster
(„in k von n Zellen“) stehen als Zählungen im Test, nicht als Urteilswort."""

import json
import math

import pytest

import fok_constants as C
import fok_conventions as CV
import fok_evaluation as V
import fok_experiments as XP
import fok_scenario as S


@pytest.fixture(scope="module")
def meas():
    return json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))


def r(x, nd=3):
    return round(x, nd)


def mean(cell, key):
    return cell[key]["mean"]


@pytest.fixture(scope="module")
def e1(meas):
    return {int(m): V.float_keys(row) for m, row in meas["e1"].items()}


@pytest.fixture(scope="module")
def e5(meas):
    return {arm: {int(m): V.float_keys(row) for m, row in per.items()} for arm, per in meas["e5"].items()}


@pytest.fixture(scope="module")
def e4(meas):
    return {k: V.int_keys(row) for k, row in meas["e4"].items()}


@pytest.fixture(scope="module")
def e3(meas):
    return {int(m): V.float_keys(row) for m, row in meas["e3"].items()}


# --- H1 und H2 -------------------------------------------------------------------------------------------------------------------------------
def test_h1_own_play_is_near_one_in_all_36_cells_and_cross_play_near_one_over_m_without_bonus(e1):
    own = [mean(c, "own") for row in e1.values() for c in row.values()]
    assert len(own) == 36 and sum(1 for x in own if x >= 0.99) == 36 and r(min(own)) == 0.995
    assert [r(mean(e1[m][0.0], "cross")) for m in C.MS] == [0.501, 0.250, 0.125, 0.063]
    assert all(abs(mean(e1[m][0.0], "cross") - 1 / m) < 0.002 for m in C.MS)          # 1/m = 0,5 / 0,25 / 0,125 / 0,0625


def test_h2_cross_play_follows_the_distribution_of_conventions_in_all_36_cells(e1):
    worst = max(abs(mean(c, "pred") - mean(c, "cross")) for row in e1.values() for c in row.values())
    assert worst < 0.001 and r(worst, 4) == 0.0008


# --- H3 und H4: der Plan-Wert und kleine Stichproben -----------------------------------------------------------------------------------------
def test_h3_the_plan_value_is_one_of_the_larger_samples_not_the_mean(meas):
    a = meas["small"]["20|6000"]["2"]["0.2"]
    b = meas["small"]["40|20000"]["2"]["0.2"]
    assert (r(a["mean"]), r(a["sd"]), r(a["p5"]), r(a["p95"]), r(a["max"]), a["ge81"]) == (0.576, 0.095, 0.479, 0.731, 0.899, 1)
    assert (r(b["mean"]), r(b["sd"]), r(b["p5"]), r(b["p95"]), r(b["max"]), b["ge81"]) == (0.577, 0.047, 0.508, 0.672, 0.672, 0)
    assert a["groups"] == b["groups"] == 100 and a["pairs_per"] == 20 and b["pairs_per"] == 40


def test_h3_the_mean_of_small_samples_agrees_with_the_large_grid_between_its_neighbours(e1, meas):
    """δ = 0,2 liegt zwischen den Gitterpunkten 0,1 und 0,3 (Fremdspiel 0,529 und 0,650 bei m = 2); das Mittel der kleinen Stichproben (0,576) liegt dazwischen."""
    lo, hi = mean(e1[2][0.1], "cross"), mean(e1[2][0.3], "cross")
    assert (r(lo), r(hi)) == (0.529, 0.650) and lo < meas["small"]["20|6000"]["2"]["0.2"]["mean"] < hi


def test_h4_small_bonuses_change_nothing_visible(e1, meas):
    excess = [mean(e1[m][d], "cross") - 1 / m for m in C.MS for d in (0.0, 0.001, 0.003, 0.01, 0.03)]
    assert len(excess) == 20 and max(excess) < 0.003 and r(max(excess), 4) == 0.0021
    s = meas["small"]
    assert [r(s["20|6000"][m]["0.05"]["mean"]) for m in ("2", "4")] == [0.512, 0.255] and [r(s["40|20000"][m]["0.05"]["mean"]) for m in ("2", "4")] == [0.502, 0.253]


# --- H5: glatte Kurve und Halbweg-Bonus -------------------------------------------------------------------------------------------------------
def test_h5_the_share_on_mark_one_rises_without_a_drop_larger_than_0_01_and_the_largest_step_is_stated(e1):
    steps = {}
    for m in C.MS:
        f = [mean(e1[m][d], "frac0") for d in C.DELTAS]
        diffs = [b - a for a, b in zip(f, f[1:])]
        assert min(diffs) > -0.01, m
        steps[m] = r(max(diffs), 2)
    assert steps == {2: 0.17, 4: 0.32, 8: 0.37, 16: 0.41}


def test_h5_the_first_bonus_with_cross_play_more_than_0_01_over_chance(e1):
    first = {m: next(d for d in C.DELTAS if mean(e1[m][d], "cross") - 1 / m > 0.01) for m in C.MS}
    assert first == {2: 0.1, 4: 0.1, 8: 0.3, 16: 0.3}


def test_h5_half_way_bonus_grows_with_m(e1):
    half = {m: CV.half_way_delta(list(e1[m]), [mean(e1[m][d], "frac0") for d in e1[m]], m) for m in C.MS}
    assert all(kind == "interpoliert" for kind, _ in half.values())
    assert [r(half[m][1], 2) for m in C.MS] == [0.25, 0.41, 0.55, 0.73]
    vals = [half[m][1] for m in C.MS]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_hook_numbers(e1):
    assert [r(mean(e1[4][d], "cross")) for d in (0.0, 0.1, 1.0)] == [0.250, 0.262, 0.750]
    assert [r(mean(e1[4][d], "own")) for d in (0.0, 0.1, 1.0)] == [0.997, 0.997, 0.999]
    assert r(mean(e1[4][0.1], "frac0")) == 0.345 and r(mean(e1[4][1.0], "frac0")) == 0.863


# --- H6: Lernverfahren statt Schrittzahl ------------------------------------------------------------------------------------------------------
def test_h6_the_half_way_bonus_depends_on_learning_rate_and_initial_spread_not_on_the_number_of_steps(meas, e1):
    def half(row):
        row = V.float_keys(row)
        return CV.half_way_delta(list(row), [mean(row[d], "frac0") for d in row], 4)[1]
    h = {k: half(v) for k, v in meas["e2"].items()}
    assert [r(h[k], 2) for k in ("0.03|0.1|6000", "0.3|0.1|6000")] == [0.21, 1.14]
    assert [r(h[k], 2) for k in ("0.1|0.0|6000", "0.1|0.5|6000")] == [0.37, 0.98]
    assert [r(h[k], 2) for k in ("0.1|0.1|2000", "0.1|0.1|20000")] == [0.43, 0.41]
    base = CV.half_way_delta(list(e1[4]), [mean(e1[4][d], "frac0") for d in e1[4]], 4)[1]
    assert r(base, 2) == 0.41
    assert max(h["0.1|0.1|2000"], h["0.1|0.1|20000"], base) / min(h["0.1|0.1|2000"], h["0.1|0.1|20000"], base) < 1.07          # Schrittzahl: Faktor unter 1,07
    assert h["0.3|0.1|6000"] / h["0.03|0.1|6000"] > 5 and h["0.1|0.5|6000"] / h["0.1|0.0|6000"] > 2.5            # Lernrate: Faktor 5,5; Streuung: Faktor 2,7


def test_h6_own_play_stays_high_in_every_sensitivity_cell(meas):
    low = min(mean(c, "own") for row in meas["e2"].values() for c in row.values())
    assert r(low) == 0.989 and low > 0.98


# --- H7: Fluss gegen Lauf ---------------------------------------------------------------------------------------------------------------------
def test_h7_the_flow_overestimates_the_bonus_in_12_of_15_cells(e3):
    over, total = 0, 0
    for m, row in e3.items():
        for d, c in row.items():
            if d > 0:
                total += 1
                over += (mean(c, "det0") - mean(c, "sto0")) > 4 * math.hypot(c["det0"]["se"], c["sto0"]["se"])
    assert (over, total) == (12, 15)
    assert [d for m in e3 for d, c in e3[m].items() if d > 0 and (mean(c, "det0") - mean(c, "sto0")) <= 4 * math.hypot(c["det0"]["se"], c["sto0"]["se"])] == [0.01, 0.01, 0.01]


def test_h7_without_bonus_the_flow_follows_the_start_favourite_and_the_run_does_not(e3):
    assert [r(mean(e3[m][0.0], "init_det")) for m in (2, 4, 8)] == [1.000, 0.976, 0.970]
    assert [r(mean(e3[m][0.0], "init_sto")) for m in (2, 4, 8)] == [0.604, 0.385, 0.256]


def test_h7_largest_gap_between_flow_and_run(e3):
    c = e3[8][0.3]
    assert (r(mean(c, "det0")), r(mean(c, "sto0"))) == (0.936, 0.360)


# --- H8: Vorerfahrung -------------------------------------------------------------------------------------------------------------------------
def test_h8_common_history_raises_cross_play_within_a_group_by_dose(e4):
    row = e4["4|0.0"]
    assert [r(mean(row[t], "within")) for t in C.E4_T0S] == [0.286, 0.431, 0.494, 0.712, 0.876, 0.994, 0.998]
    assert all(b > a for a, b in zip([mean(row[t], "within") for t in C.E4_T0S], [mean(row[t], "within") for t in C.E4_T0S][1:]))


def test_h8_between_groups_stays_at_chance(e4):
    for m in C.E4_MS:
        lo, hi = (min(mean(c, "between") for c in e4[f"{m}|0.0"].values()), max(mean(c, "between") for c in e4[f"{m}|0.0"].values()))
        assert (r(lo), r(hi)) == {2: (0.493, 0.508), 4: (0.240, 0.257), 8: (0.124, 0.131)}[m]
        assert abs(lo - 1 / m) < 0.01 and abs(hi - 1 / m) < 0.01


def test_h8_the_reference_does_not_need_to_be_finished(e4):
    c = e4["4|0.0"][400]
    assert (r(mean(c, "ref_own")), r(mean(c, "within")), r(mean(c, "between"))) == (0.829, 0.994, 0.246)


def test_h8_longer_history_is_needed_for_more_places(e4):
    first = {m: next(t for t in C.E4_T0S if mean(e4[f"{m}|0.0"][t], "within") >= 0.9) for m in C.E4_MS}
    assert first == {2: 100, 4: 400, 8: 1000}
    assert [r(mean(e4[f"{m}|0.0"][400], "within")) for m in C.E4_MS] == [0.998, 0.994, 0.737]


def test_h8_a_common_random_start_lifts_cross_play_slightly(e4):
    assert [(r(mean(e4[f"{m}|0.0"][0], "within")), r(mean(e4[f"{m}|0.0"][0], "between"))) for m in C.E4_MS] == [(0.536, 0.501), (0.286, 0.249), (0.151, 0.125)]


def test_h8_a_small_bonus_changes_little_with_history(e4):
    assert [r(mean(e4["4|0.05"][t], "within")) for t in (0, 100, 400)] == [0.299, 0.719, 0.995]


def test_h8_communities_row(meas):
    comm = {int(g): c for g, c in meas["e4comm"].items()}
    assert [r(comm[g]["500"]["within"]["mean"]) for g in C.GROUP_SIZES] == [0.913, 0.867, 0.800, 0.705, 0.595]
    assert [r(comm[g]["500"]["sure"]["mean"], 3) for g in C.GROUP_SIZES] == [0.841, 0.599, 0.359, 0.129, 0.031]
    assert min(comm[g]["2000"]["within"]["mean"] for g in C.GROUP_SIZES) > 0.988 and r(min(comm[g]["2000"]["within"]["mean"] for g in C.GROUP_SIZES)) == 0.989
    assert all(abs(comm[g][t]["between"]["mean"] - 0.25) < 0.003 for g in C.GROUP_SIZES for t in ("500", "1000", "2000"))
    assert comm[2]["communities"] == 4 * C.GROUP_COUNT


# --- H9 und H10: Other-Play ------------------------------------------------------------------------------------------------------------------
def _half(row, m):
    return CV.half_way_delta(C.OP_DELTAS, [mean(row[d], "frac0") for d in C.OP_DELTAS], m)[1]


def test_h9_platz_1_fest_lowers_the_half_way_bonus_for_four_or_more_places(e5):
    own = {m: _half(e5["Selbstspiel"][m], m) for m in (4, 8, 16)}
    fix = {m: _half(e5["Platz 1 fest"][m], m) for m in (4, 8, 16)}
    assert [r(own[m], 2) for m in (4, 8, 16)] == [0.40, 0.53, 0.75] and [r(fix[m], 2) for m in (4, 8, 16)] == [0.15, 0.12, 0.14]
    assert all(fix[m] < own[m] / 2 for m in (4, 8, 16))


def test_h9_cross_play_with_platz_1_fest_is_higher_in_21_of_21_cells(e5):
    count = sum(mean(e5["Platz 1 fest"][m][d], "cross") > mean(e5["Selbstspiel"][m][d], "cross") for m in (4, 8, 16) for d in C.OP_DELTAS)
    assert count == 21
    assert [(r(mean(e5["Selbstspiel"][m][0.3], "cross")), r(mean(e5["Platz 1 fest"][m][0.3], "cross"))) for m in (4, 8, 16)] == [(0.368, 0.578), (0.202, 0.550), (0.100, 0.475)]


def test_h9_the_price_is_own_play(e5):
    assert [r(mean(e5["Platz 1 fest"][m][0.0], "own")) for m in (4, 8, 16)] == [0.639, 0.475, 0.346]
    assert [r(mean(e5["Platz 1 fest"][m][1.0], "own")) for m in (4, 8, 16)] == [0.973, 0.964, 0.940]
    assert [r(mean(e5["Platz 1 fest"][m][0.3], "own")) for m in (4, 8)] == [0.823, 0.767] and r(mean(e5["Selbstspiel"][8][0.3], "own")) == 0.997
    assert min(mean(c, "own") for m in C.MS for c in e5["Selbstspiel"][m].values()) > 0.99


def test_h9_the_group_itself_sets_a_focal_point_without_bonus(e5):
    assert [r(mean(e5["Platz 1 fest"][m][0.0], "cross")) for m in (4, 8, 16)] == [0.302, 0.209, 0.138]
    assert [r(mean(e5["Selbstspiel"][m][0.0], "cross")) for m in (4, 8, 16)] == [0.251, 0.126, 0.062]


def test_h10_the_wrong_group_gives_no_coordination_in_28_of_28_cells(e5):
    cells = [(m, d) for m in C.MS for d in C.OP_DELTAS]
    near = [(m, d) for m, d in cells if abs(mean(e5["alle Plätze"][m][d], "own") - 1 / m) <= 0.016 and abs(mean(e5["alle Plätze"][m][d], "cross") - 1 / m) <= 0.013]
    assert len(near) == len(cells) == 28
    assert [r(mean(e5["alle Plätze"][m][0.3], "own")) for m in (4, 8)] == [0.251, 0.127] and [r(mean(e5["alle Plätze"][m][0.3], "frac0")) for m in (4, 8)] == [0.955, 0.864]


def test_h10_platz_1_fest_is_the_same_mechanism_as_selbstspiel_for_two_places(e5):
    diffs = [abs(mean(e5["Platz 1 fest"][2][d], "cross") - mean(e5["Selbstspiel"][2][d], "cross")) for d in C.OP_DELTAS]
    assert len(diffs) == 7 and max(diffs) < 0.015 and r(max(diffs)) == 0.014


# --- Hilfetexte der Presets -----------------------------------------------------------------------------------------------------------------
def test_preset_help_numbers(e1, e4, e5):
    h = C.PRESET_HELP
    assert "0,997" in h["Standard (m = 4, ohne Bonus)"] and "0,250" in h["Standard (m = 4, ohne Bonus)"] and (r(mean(e1[4][0.0], "own")), r(mean(e1[4][0.0], "cross"))) == (0.997, 0.250)
    assert "0,995" in h["Fremde Paare (m = 16)"] and "0,063" in h["Fremde Paare (m = 16)"] and (r(mean(e1[16][0.0], "own")), r(mean(e1[16][0.0], "cross"))) == (0.995, 0.063)
    assert "0,250 auf 0,262" in h["Kleiner Bonus (δ = 0,1)"] and r(mean(e1[4][0.1], "cross")) == 0.262
    assert "0,750" in h["Großer Bonus (δ = 1)"] and "0,863" in h["Großer Bonus (δ = 1)"] and (r(mean(e1[4][1.0], "cross")), r(mean(e1[4][1.0], "frac0"))) == (0.750, 0.863)
    g = h["Gleiche Vorgeschichte (T₀ = 400)"]
    assert "0,994" in g and "0,246" in g and "0,829" in g and "400 Schritte" in g
    f = h["Other-Play, Platz 1 fest (m = 8, δ = 0,3)"]
    assert all(x in f for x in ("0,550", "0,202", "0,767", "0,997")) and (r(mean(e5["Platz 1 fest"][8][0.3], "cross")), r(mean(e5["Selbstspiel"][8][0.3], "cross"))) == (0.550, 0.202)
    w = h["Other-Play, falsche Gruppe (m = 4, δ = 0,3)"]
    assert "0,955" in w and "0,251" in w and r(mean(e5["alle Plätze"][4][0.3], "frac0")) == 0.955


def test_the_hook_in_the_help_and_the_ten_percent_statement(e1):
    """„10 % mehr Zeit“ ist δ = 0,1: Platz 1 bringt 1,1 statt 1."""
    assert C.PRESETS["Kleiner Bonus (δ = 0,1)"]["delta"] == 0.1 and C.PRESETS["Großer Bonus (δ = 1)"]["delta"] == 1.0


# --- exakte Zahlen ----------------------------------------------------------------------------------------------------------------------------
def test_null_standard_deviation_at_the_default_settings():
    assert r(CV.cross_null_sd(4, 300), 4) == 0.0020 and r(CV.cross_null_sd(2, 20), 4) == 0.0372
    assert S.budget_pairs(S.Settings(pairs=500, steps=20_000)) == 150 and S.budget_pairs(S.Settings(pairs=500, steps=6000)) == 500


def test_the_default_cell_has_300_pairs_and_the_slider_maximum_500():
    assert (C.DEFAULT_PAIRS, C.PAIRS_MAX, C.DEFAULT_STEPS, C.STEPS_MAX, C.PAIRS_MIN) == (300, 500, 6000, 20000, 1)
