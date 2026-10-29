"""Experimente E1 bis E5: Zellen und Tabellen aus dem Kern (live, mit wenigen Paaren) und die vorgerechneten Messreihen (data/measurements.json, mit vielen Paaren).

Eine Zelle sind `blocks` unabhängige Blöcke von Paaren (je ein Trainingslauf mit eigenem SplitMix64-Seed aus der Zellenbezeichnung); ausgewiesen werden Mittel und Standardfehler der Blockwerte.
Die vorgerechneten Reihen (tools/make_measurements.py) benutzen genau diese Funktionen mit C.MEAS_BLOCKS Blöcken zu je C.MEAS_PAIRS Paaren und tragen den Hash der Dateien, aus deren Code sie stammen."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import fok_constants as C
import fok_conventions as CV
import fok_experience as EX
import fok_learning as L
import fok_policy as PO
import fok_scenario as S

ROOT = Path(__file__).resolve().parent
MEAS_FILE = ROOT / "data" / "measurements.json"
HASHED_FILES = ("fok_policy.py", "fok_symmetry.py", "fok_learning.py", "fok_experience.py", "fok_conventions.py", "fok_scenario.py", "fok_experiments.py", "fok_rng.py")
ARMS = {"Selbstspiel": C.MODE_SELF, "alle Plätze": C.MODE_ALL, "Platz 1 fest": C.MODE_FIX0}


# --------------------------------------------------------------------------------- Hilfen
def seed_of(label: str) -> int:
    """Zufalls-Seed aus einer Zellenbezeichnung (FNV-1a über die UTF-8-Bytes; Python-`hash` ist je Prozess gesalzen und taugt nicht)."""
    h = 0xCBF29CE484222325
    for byte in label.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % (C.SEED_MAX + 1)


def stats(blocks: list[dict]) -> dict:
    """Mittel und Standardfehler (Standardabweichung der Blockwerte durch √Blöcke) je Kennzahl; fehlt ein Wert in einem Block (None), ist das Mittel None."""
    out: dict = {"blocks": len(blocks)}
    for key in blocks[0]:
        vals = [b[key] for b in blocks]
        if any(v is None for v in vals):
            out[key] = {"mean": None, "se": None}
            continue
        arr = np.array(vals, dtype=float)
        out[key] = {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0}
    return out


def base_settings(label: str, **fields) -> S.Settings:
    return S.Settings(seed=seed_of(label), **fields)


# --------------------------------------------------------------------------------- E1, E2, E5: Zellen aus Trainingsläufen
def cell(s: S.Settings, blocks: int, label: str) -> dict:
    """Eine Zelle: `blocks` Trainingsläufe mit s.pairs Paaren, Seed des Blocks b = seed_of(f"{label}|b{b}")."""
    runs = []
    for b in range(blocks):
        sb = S.with_(s, seed=seed_of(f"{label}|b{b}"))
        runs.append(CV.block_metrics(L.train_pairs(sb).theta))
    out = stats(runs)
    out["pairs"] = blocks * s.pairs
    return out


def delta_row(s: S.Settings, blocks: int, label: str, deltas=C.DELTAS) -> dict:
    """Eine Zeile über den Bonus: {δ: Zelle}."""
    return {d: cell(S.with_(s, delta=d), blocks, f"{label}|d{d}") for d in deltas}


def e1_live(s: S.Settings, ms=C.MS, deltas=(0.0, 0.1, 1.0)) -> dict:
    """E1 mit den Reglern der Seitenleiste (Lernrate, Streuung, Schritte, Paare, Modus): {m: {δ: Zelle}} mit einem Block."""
    return {m: delta_row(S.with_(s, m=m, t0=0), 1, f"e1live|{s.seed}|{m}", deltas) for m in ms}


# --------------------------------------------------------------------------------- E2: Streuung kleiner Stichproben
def small_sample(m: int, delta: float, pairs_per: int, steps: int, groups: int, label: str) -> dict:
    """`groups` unabhängige Stichproben zu je `pairs_per` Paaren (ein Trainingslauf mit groups · pairs_per Paaren, in Gruppen geteilt): Fremdspiel je Stichprobe, davon Mittel, Standardabweichung,
    5.–95. Perzentil, Kleinstes, Größtes. `ge81`: Zahl der Stichproben mit Fremdspiel ≥ 0,81 (der Plan-Wert, siehe README)."""
    s = base_settings(label, m=m, delta=delta, pairs=groups * pairs_per, steps=steps)
    p = PO.softmax(L.train_pairs(s).theta)
    pa, pb = p[:, 0], p[:, 1]
    vals = []
    own = []
    for g in range(groups):
        sl = slice(g * pairs_per, (g + 1) * pairs_per)
        vals.append(PO.cross_play(pa[sl], pb[sl]))
        own.append(PO.own_play(pa[sl], pb[sl]))
    v = np.array(vals)
    return {"m": m, "delta": delta, "pairs_per": pairs_per, "steps": steps, "groups": groups, "values": [float(x) for x in v], "own": float(np.mean(own)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max()), "ge81": int((v >= 0.81).sum())}


# --------------------------------------------------------------------------------- E3: Erwartungsfluss gegen Lauf
def flow_cell(s: S.Settings, blocks: int, label: str) -> dict:
    """Erwartungsfluss und Lauf vom selben Start: Anteil der Paare, deren Siegermarke (argmax p_A + p_B) im Fluss / im Lauf / am Start Marke 0 ist, Übereinstimmung Fluss = Lauf,
    Start-Favorit = Fluss, Start-Favorit = Lauf, Eigenspiel im Fluss."""
    runs = []
    for b in range(blocks):
        sb = S.with_(s, seed=seed_of(f"{label}|b{b}"), mode=C.MODE_SELF, t0=0)
        stream = L.Stream(sb.seed)
        th0 = L.initial_theta(sb.pairs, sb.m, sb.sigma0, stream)
        det = L.expected_flow(th0, sb.delta, sb.steps, sb.eta)
        sto = L.train_pairs(sb, theta0=th0, stream=stream).theta

        def win(th):
            p = PO.softmax(th)
            return CV.convention_labels(p[:, 0], p[:, 1])

        w_det, w_sto, w_init = win(det), win(sto), win(th0)
        pd = PO.softmax(det)
        runs.append({"det0": float((w_det == 0).mean()), "sto0": float((w_sto == 0).mean()), "init0": float((w_init == 0).mean()), "agree": float((w_det == w_sto).mean()),
                     "init_det": float((w_init == w_det).mean()), "init_sto": float((w_init == w_sto).mean()), "det_own": PO.own_play(pd[:, 0], pd[:, 1])})
    out = stats(runs)
    out["pairs"] = blocks * s.pairs
    return out


# --------------------------------------------------------------------------------- E4: Vorerfahrung
def reference_cell(s: S.Settings, refs: int, per_ref: int, t0: int, blocks: int, label: str) -> dict:
    """Gemeinsame Referenz: `refs` Referenzen lernen t0 Schritte, je `per_ref` Paare starten von ihr und lernen s.steps Schritte mit eigenem Partner weiter.
    Fremdspiel innerhalb einer Gruppe (gleiche Vorgeschichte) und zwischen Gruppen, Eigenspiel, Anteil der Paare, die bei der Konvention der Referenz landen, Eigenspiel der Referenz selbst."""
    runs = []
    for b in range(blocks):
        sb = S.with_(s, seed=seed_of(f"{label}|b{b}"), pairs=refs * per_ref)
        stream = L.Stream(sb.seed)
        start, group = EX.shared_reference(sb, refs, per_ref, stream, t0=t0)
        th = L.train_pairs(sb, theta0=start, stream=stream).theta
        p, p0 = PO.softmax(th), PO.softmax(start)
        within, between = PO.cross_play_groups(p[:, 0], p[:, 1], group)
        follow = float((CV.convention_labels(p[:, 0], p[:, 1]) == CV.convention_labels(p0[:, 0], p0[:, 1])).mean())
        runs.append({"within": within, "between": between, "own": PO.own_play(p[:, 0], p[:, 1]), "follow": follow, "ref_own": PO.own_play(p0[:, 0], p0[:, 1]),
                     "ref_pmax": float(((p0[:, 0] + p0[:, 1]) / 2).max(axis=1).mean())})
    out = stats(runs)
    out["pairs"] = blocks * refs * per_ref
    return out


def e4_live(s: S.Settings, t0s=C.E4_T0S, refs: int = C.E4_LIVE_REFS, per_ref: int = C.E4_LIVE_PER_REF) -> dict:
    """E4 mit den Reglern der Seitenleiste (m, δ, Lernrate, Streuung, Schritte): {T₀: Zelle} mit einem Block aus refs · per_ref Paaren."""
    base = S.with_(s, mode=C.MODE_SELF, t0=0)
    return {t0: reference_cell(base, refs, per_ref, t0, 1, f"e4live|{s.seed}|{s.m}|{s.delta}|{t0}") for t0 in t0s}


def community_cell(m: int, size: int, communities: int, blocks: int, label: str, checkpoints=C.GROUP_ROUNDS) -> dict:
    """Gemeinschaften (siehe fok_experience.community): Mittel und Standardfehler je Haltepunkt über `blocks` Blöcke zu je `communities` Gemeinschaften (δ = 0, η = 0,1, σ₀ = 0,1)."""
    per_block = []
    for b in range(blocks):
        stream = L.Stream(seed_of(f"{label}|b{b}"))
        per_block.append(EX.community(m, size, communities, max(checkpoints), C.DEFAULT_ETA, 0.0, C.DEFAULT_SIGMA, stream, tuple(checkpoints)))
    out: dict = {"size": size, "communities": blocks * communities}
    for t in checkpoints:
        out[str(t)] = stats([pb[t] for pb in per_block])
    return out


# --------------------------------------------------------------------------------- vorgerechnete Reihen
def generator_hash() -> str:
    """Hash der Dateien, aus deren Code die Messreihen stammen (Zeilenenden normalisiert, damit CRLF/LF denselben Hash geben)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0" + text + b"\0")
    return h.hexdigest()[:16]


def to_jsonable(obj):
    """Wörterbücher mit Zahlen- oder Tupelschlüsseln in JSON-fähige (Text-)Schlüssel umwandeln."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_measurements() -> dict | None:
    """Die vorgerechneten Reihen, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört (dann zeigt die App sie nicht)."""
    if not MEAS_FILE.exists():
        return None
    data = json.loads(MEAS_FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


def write_measurements(payload: dict) -> None:
    out = {"generator_hash": generator_hash(), **to_jsonable(payload)}
    MEAS_FILE.parent.mkdir(exist_ok=True)
    MEAS_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------- Aufträge der vorgerechneten Reihen (ein Auftrag = eine Zelle, in Prozessen rechenbar)
def jobs() -> list[tuple]:
    """Alle Zellen der vorgerechneten Reihen als Aufträge (Art, Schlüssel, Argumente); `run_job` rechnet einen Auftrag. Reihenfolge fest."""
    out: list[tuple] = []
    B, P = C.MEAS_BLOCKS, C.MEAS_PAIRS
    for m in C.MS:
        for d in C.DELTAS:
            out.append(("cell", ("e1", m, d), dict(m=m, delta=d, pairs=P), B, f"e1|{m}|{d}"))
    configs = [(0.03, 0.1, 6000), (0.3, 0.1, 6000), (0.1, 0.0, 6000), (0.1, 0.5, 6000), (0.1, 0.1, 2000), (0.1, 0.1, 20000)]
    for eta, sigma, steps in configs:
        for d in C.DELTAS:
            out.append(("cell", ("e2", f"{eta}|{sigma}|{steps}", d), dict(m=4, delta=d, pairs=P, eta=eta, sigma0=sigma, steps=steps), B, f"e2|{eta}|{sigma}|{steps}|{d}"))
    for pairs_per, steps in C.SMALL_SETUPS:
        for d in C.SMALL_DELTAS:
            for m in (2, 4):
                out.append(("small", ("small", f"{pairs_per}|{steps}", m, d), (m, d, pairs_per, steps, C.SMALL_GROUPS), f"small|{pairs_per}|{steps}|{m}|{d}"))
    for m in C.E3_MS:
        for d in C.E3_DELTAS:
            out.append(("flow", ("e3", m, d), dict(m=m, delta=d, pairs=P), B, f"e3|{m}|{d}"))
    for m in C.E4_MS:
        for t0 in C.E4_T0S:
            out.append(("ref", ("e4", m, 0.0, t0), dict(m=m, delta=0.0), (C.E4_REFS, C.E4_PER_REF, t0), B, f"e4|{m}|0.0|{t0}"))
    for t0 in (0, 100, 400):
        out.append(("ref", ("e4", 4, 0.05, t0), dict(m=4, delta=0.05), (C.E4_REFS, C.E4_PER_REF, t0), B, f"e4|4|0.05|{t0}"))
    for g in C.GROUP_SIZES:
        out.append(("comm", ("e4comm", g), (4, g, C.GROUP_COUNT, B), f"e4comm|{g}"))
    for arm, mode in ARMS.items():
        for m in C.MS:
            for d in C.OP_DELTAS:
                out.append(("cell", ("e5", arm, m, d), dict(m=m, delta=d, pairs=P, mode=mode), B, f"e5|{arm}|{m}|{d}"))
    return out


def run_job(job: tuple):
    """Rechnet einen Auftrag aus `jobs()`; Rückgabe (Schlüssel, Ergebnis)."""
    kind, key = job[0], job[1]
    if kind == "cell":
        _, _, fields, blocks, label = job
        return key, cell(S.Settings(**fields), blocks, label)
    if kind == "flow":
        _, _, fields, blocks, label = job
        return key, flow_cell(S.Settings(**fields), blocks, label)
    if kind == "ref":
        _, _, fields, (refs, per_ref, t0), blocks, label = job
        return key, reference_cell(S.Settings(**fields), refs, per_ref, t0, blocks, label)
    if kind == "small":
        _, _, (m, d, pairs_per, steps, groups), label = job
        return key, small_sample(m, d, pairs_per, steps, groups, label)
    if kind == "comm":
        _, _, (m, g, comms, blocks), label = job
        return key, community_cell(m, g, comms, blocks, label)
    raise ValueError(kind)


def assemble(results: dict) -> dict:
    """Ordnet die Ergebnisse der Aufträge (Schlüssel → Ergebnis) zu dem Wörterbuch, das in data/measurements.json steht."""
    out: dict = {"blocks": C.MEAS_BLOCKS, "pairs_per_block": C.MEAS_PAIRS, "e1": {}, "e2": {}, "small": {}, "e3": {}, "e4": {}, "e4comm": {}, "e5": {}}
    for job in jobs():                                     # feste Reihenfolge der Aufträge, unabhängig davon, in welcher Reihenfolge die Prozesse fertig wurden
        key, val = job[1], results[job[1]]
        tag = key[0]
        if tag == "e1":
            out["e1"].setdefault(key[1], {})[key[2]] = val
        elif tag == "e2":
            out["e2"].setdefault(key[1], {})[key[2]] = val
        elif tag == "small":
            out["small"].setdefault(key[1], {}).setdefault(key[2], {})[key[3]] = val
        elif tag == "e3":
            out["e3"].setdefault(key[1], {})[key[2]] = val
        elif tag == "e4":
            out["e4"].setdefault(f"{key[1]}|{key[2]}", {})[key[3]] = val
        elif tag == "e4comm":
            out["e4comm"][key[1]] = val
        elif tag == "e5":
            out["e5"].setdefault(key[1], {}).setdefault(key[2], {})[key[3]] = val
    return out
