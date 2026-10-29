"""Die vorgerechneten Messreihen (data/measurements.json) gehören zum aktuellen Rechencode und werden von der Rechnung reproduziert.

Eigene Datei, weil jede Änderung an fok_policy, fok_symmetry, fok_learning, fok_experience, fok_conventions, fok_scenario, fok_experiments oder fok_rng den Erzeuger-Hash ändert: so bleibt die Meldung
„Messdatei neu erzeugen“ von den Tests der Rechnung getrennt. Reproduziert wird plattformsicher: Zellen mit anderen Seeds (anderen Zellenbezeichnungen) innerhalb einer Marge, die aus der
Streuung der Blöcke und einer festen Untergrenze folgt; nichts an einzelnen Läufen."""

import json
import math

import pytest

import fok_constants as C
import fok_experiments as XP
import fok_scenario as S


@pytest.fixture(scope="module")
def stored():
    return json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash(stored):
    assert stored["generator_hash"] == XP.generator_hash(), "fok_policy/symmetry/learning/experience/conventions/scenario/experiments/rng wurde geändert: python tools/make_measurements.py ausführen"
    assert XP.load_measurements() is not None


def test_load_returns_none_for_a_stale_hash_and_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(XP, "generator_hash", lambda: "0000000000000000")
    assert XP.load_measurements() is None
    monkeypatch.undo()
    monkeypatch.setattr(XP, "MEAS_FILE", tmp_path / "fehlt.json")
    assert XP.load_measurements() is None


def test_axes_and_sizes_match_the_code(stored):
    assert stored["blocks"] == C.MEAS_BLOCKS and stored["pairs_per_block"] == C.MEAS_PAIRS
    assert list(stored["e1"]) == [str(m) for m in C.MS]
    assert all(list(row) == [str(d) for d in C.DELTAS] for row in stored["e1"].values())
    assert len(stored["e2"]) == 6 and all(list(row) == [str(d) for d in C.DELTAS] for row in stored["e2"].values())
    assert list(stored["e3"]) == [str(m) for m in C.E3_MS] and all(list(row) == [str(d) for d in C.E3_DELTAS] for row in stored["e3"].values())
    assert list(stored["e4"]) == [f"{m}|0.0" for m in C.E4_MS] + ["4|0.05"]
    assert all(list(stored["e4"][f"{m}|0.0"]) == [str(t) for t in C.E4_T0S] for m in C.E4_MS) and list(stored["e4"]["4|0.05"]) == ["0", "100", "400"]
    assert list(stored["e4comm"]) == [str(g) for g in C.GROUP_SIZES]
    assert list(stored["e5"]) == list(XP.ARMS)
    assert all(list(stored["e5"][arm]) == [str(m) for m in C.MS] and all(list(r) == [str(d) for d in C.OP_DELTAS] for r in stored["e5"][arm].values()) for arm in XP.ARMS)
    assert list(stored["small"]) == [f"{p}|{t}" for p, t in C.SMALL_SETUPS]
    assert all(c["blocks"] == C.MEAS_BLOCKS for row in stored["e1"].values() for c in row.values()) and stored["e1"]["4"]["0.0"]["pairs"] == C.MEAS_BLOCKS * C.MEAS_PAIRS


def walk(obj):
    """Alle Kennzahl-Einträge {"mean", "se"} der Datei."""
    if isinstance(obj, dict):
        if set(obj) == {"mean", "se"}:
            yield obj
        else:
            for v in obj.values():
                yield from walk(v)


def test_every_stored_statistic_is_a_finite_number_in_the_unit_interval_with_a_nonnegative_error(stored):
    count = 0
    for st in walk({k: v for k, v in stored.items() if k != "small"}):
        count += 1
        assert st["mean"] is not None and 0.0 <= st["mean"] <= 1.0 and st["se"] >= 0.0 and math.isfinite(st["mean"]) and math.isfinite(st["se"]), st
    assert count > 1000


def test_every_e1_cell_has_a_defined_cross_play_and_no_cell_is_all_zero(stored):
    """Nullspalten-Regel: keine Spalte mit lauter Nullen (die Variante liefe nie)."""
    for key in ("own", "cross", "frac0", "pred"):
        vals = [c[key]["mean"] for row in stored["e1"].values() for c in row.values()]
        assert max(vals) > 0.05 and len(set(round(v, 3) for v in vals)) > 5, key
    assert min(c[k]["mean"] for row in stored["e1"].values() for c in row.values() for k in ("sure", "pmax")) >= 0.99          # im Selbstspiel legt sich jedes Paar fest: die Spalte ist 1 und kein toter Zweig


def test_small_sample_values_are_consistent_with_their_summary(stored):
    for setup in stored["small"].values():
        for per_m in setup.values():
            for c in per_m.values():
                v = c["values"]
                assert len(v) == c["groups"] == C.SMALL_GROUPS and abs(sum(v) / len(v) - c["mean"]) < 1e-12 and c["min"] == min(v) and c["max"] == max(v)
                assert c["ge81"] == sum(1 for x in v if x >= 0.81) and c["p5"] <= c["p95"] and c["min"] <= c["p5"] and c["p95"] <= c["max"]


def _fresh_agrees(stored_cell, fresh_cell, keys, floor=0.04):
    for key in keys:
        a, b = stored_cell[key], fresh_cell[key]
        tol = max(4 * math.hypot(a["se"], b["se"]), floor)
        assert abs(a["mean"] - b["mean"]) < tol, (key, a["mean"], b["mean"], tol)


def test_an_e1_cell_is_reproduced_with_other_seeds(stored):
    fresh = XP.cell(S.Settings(m=4, delta=1.0, pairs=C.MEAS_PAIRS), C.MEAS_BLOCKS, "neu|e1|4|1.0")
    _fresh_agrees(stored["e1"]["4"]["1.0"], fresh, ("own", "cross", "frac0"))


def test_an_other_play_cell_is_reproduced_with_other_seeds(stored):
    fresh = XP.cell(S.Settings(m=8, delta=0.3, pairs=C.MEAS_PAIRS, mode=C.MODE_FIX0), C.MEAS_BLOCKS, "neu|e5|fest|8|0.3")
    _fresh_agrees(stored["e5"]["Platz 1 fest"]["8"]["0.3"], fresh, ("own", "cross", "frac0"), floor=0.05)


def test_a_flow_cell_is_reproduced_with_other_seeds(stored):
    fresh = XP.flow_cell(S.Settings(m=4, delta=0.3, pairs=C.MEAS_PAIRS), C.MEAS_BLOCKS, "neu|e3|4|0.3")
    _fresh_agrees(stored["e3"]["4"]["0.3"], fresh, ("det0", "sto0", "init_det"))


def test_a_reference_cell_is_reproduced_with_other_seeds(stored):
    fresh = XP.reference_cell(S.Settings(m=4, delta=0.0), C.E4_REFS, C.E4_PER_REF, 400, C.MEAS_BLOCKS, "neu|e4|4|400")
    _fresh_agrees(stored["e4"]["4|0.0"]["400"], fresh, ("within", "between"), floor=0.05)


def test_assemble_is_independent_of_the_order_in_which_the_results_arrive():
    jobs = XP.jobs()
    results = {j[1]: {"mark": i} for i, j in enumerate(jobs)}
    shuffled = dict(reversed(list(results.items())))
    a, b = XP.assemble(results), XP.assemble(shuffled)
    assert json.dumps(a) == json.dumps(b)


def test_every_job_has_a_distinct_key_and_a_distinct_seed_label():
    jobs = XP.jobs()
    assert len({j[1] for j in jobs}) == len(jobs) == 237
    labels = [j[-1] for j in jobs]
    assert len(set(labels)) == len(labels)
