"""Rechnet alle vorgerechneten Messreihen (Bonus-Gitter, Empfindlichkeit, kleine Stichproben, Erwartungsfluss, Vorerfahrung, Gemeinschaften, Other-Play) neu und schreibt
data/measurements.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_measurements.py [--workers 8]
Ändert sich eine der Dateien fok_policy.py, fok_symmetry.py, fok_learning.py, fok_experience.py, fok_conventions.py, fok_scenario.py, fok_experiments.py oder fok_rng.py, stimmt der Hash
nicht mehr (tests/test_measurements.py schlägt an, die App zeigt die vorgerechneten Reihen dann nicht); danach diesen Aufruf wiederholen. Jede Zelle hat feste Zufalls-Seeds aus ihrer
Bezeichnung (fok_experiments.seed_of), die Reihen sind also bei gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen und anderer Reihenfolge der Ergebnisse.
Dauer: einige Minuten (237 Aufträge, verteilt auf die Prozesse)."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fok_experiments as XP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start = time.time()
    jobs = XP.jobs()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(XP.run_job, j) for j in jobs]
        for done, f in enumerate(as_completed(futures), 1):
            key, res = f.result()
            results[key] = res
            if done % 10 == 0 or done == len(jobs):
                print(f"[{time.time() - start:6.1f} s] {done}/{len(jobs)} Aufträge", flush=True)
    payload = XP.assemble(results)
    XP.write_measurements(payload)
    print(f"geschrieben: {XP.MEAS_FILE}  ({time.time() - start:.0f} s, Hash {XP.generator_hash()}, {len(jobs)} Aufträge, {payload['blocks']} Blöcke zu je {payload['pairs_per_block']} Paaren)")


if __name__ == "__main__":
    main()
