"""Unit tests for attacks: output parses, behaviour is unchanged, text changed.

The heavy check (every canonical EvalPlus solution still passes the plus
tests after every attack) is in scripts/04_attack.py --canonical.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.attacks import ATTACKS, AttackError, apply_attack, rename, strip_comments  # noqa: E402

SAMPLES = [
    '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers are closer than threshold."""
    # brute force
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):  # inner loop
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
    '''def f(xs, key=None):
    """Doc
    spanning lines."""
    total = 0
    msg = """multi
line"""
    for x in sorted(xs, key=key):
        total += x
    return total, len(msg)
''',
    '''import math
class A:
    k = 3
    def g(self, n):
        return [i * self.k for i in range(n)]

def solve(n):
    a = A()
    s = sum(a.g(n))
    return math.floor(s / 2)
''',
]

CALLS = [
    ("has_close_elements", [([1.0, 2.0, 2.05], 0.1), ([1.0, 5.0], 0.5)]),
    ("f", [([3, 1, 2],), ([5, -1],)]),
    ("solve", [(4,), (10,)]),
]


def _run(src, fn, args_list):
    ns = {}
    exec(compile(src, "<t>", "exec"), ns)
    return [ns[fn](*a) for a in args_list]


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("i", range(len(SAMPLES)))
def test_attack_preserves_behaviour(attack, i):
    src = SAMPLES[i]
    fn, args = CALLS[i]
    try:
        out = apply_attack(attack, src, salt="t")
    except AttackError:
        pytest.skip("attack not applicable")
    assert _run(out, fn, args) == _run(src, fn, args)
    assert fn in out  # entry point name never renamed


def test_strip_comments_removes_all():
    out = strip_comments(SAMPLES[0])
    assert "#" not in out and '"""' not in out


def test_strip_comments_noop_without_comments():
    src = "def f(x):\n    return x + 1\n"
    assert strip_comments(src) == src


def test_rename_counts():
    src = SAMPLES[0]
    names = ("numbers", "threshold", "idx", "elem", "distance")
    def kept(out):
        body = strip_comments(out).replace("has_close_elements", "")
        return sum(n in body for n in names)
    assert kept(rename(src, 1, "t")) == len(names) - 1
    assert kept(rename(src, 2, "t")) == len(names) - 2
    allr = strip_comments(rename(src, None, "t"))
    for name in ("numbers", "threshold", "idx", "elem", "distance"):
        assert name not in allr.replace("has_close_elements", "")


def test_rename_skips_keyword_param():
    # `key` is used as a keyword argument, so it must not be renamed
    out = rename(SAMPLES[1], None)
    assert "key=" in out
    assert "def f(" in out and "key=None" in out


def test_deadcode_changes_text():
    out = apply_attack("deadcode_try", SAMPLES[0], salt="t")
    assert "try:" in out and out != SAMPLES[0]


def test_rename_names_vary_by_sample():
    a, b = rename(SAMPLES[0], None, "task/1"), rename(SAMPLES[0], None, "task/2")
    assert a != b
    assert rename(SAMPLES[0], None, "task/1") == a  # deterministic
    assert "var_" not in a
