"""SplitMix64-Kopie: wörtlich wie im Original, bekannte Werte, Block gleich Strom, keine anderen Importe."""

import ast
from pathlib import Path

import numpy as np
import pytest

from fok_rng import SplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "fok_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "fok_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_uniforms_are_in_the_unit_interval_and_roughly_uniform():
    u = uniform_block(99, 200_000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 4 * (1 / 12) ** 0.5 / 200_000 ** 0.5
