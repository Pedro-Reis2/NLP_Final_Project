"""Multi-channel watermark (src/wm/multichannel.py): code, embedding round trip, detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm import multichannel as mc  # noqa: E402

PROG = '''
def solve(values, limit):
    total_sum = 0
    seen_items = []
    best_value = None
    for item_value in values:
        if item_value > limit:
            continue
        if best_value is None or item_value < best_value:
            best_value = item_value
        total_sum = total_sum + item_value
        seen_items += [item_value]
        if len(seen_items) == 3:
            break
    lookup_table = dict()
    lookup_table[0] = total_sum
    return (total_sum, best_value, seen_items, lookup_table)
'''


def _run(src, *args):
    ns = {}
    exec(src, ns)
    return ns["solve"](*args)


def test_code_corrects_erasures_and_rejects_errors():
    assert mc.decode(mc.CODEWORD) == mc.payload_bits()
    w = list(mc.CODEWORD)
    w[0] = w[17] = w[31] = None
    assert mc.decode(w) == mc.payload_bits()
    w = list(mc.CODEWORD)
    w[5] ^= 1
    assert mc.decode(w) != mc.payload_bits()


def test_embed_preserves_behaviour_and_roundtrips():
    out = mc.embed(PROG)
    for args in ([5, 1, 9, 3, 7], 6), ([], 1), ([2, 2, 2, 2], 2):
        assert _run(out, *args) == _run(PROG, *args)
    assert mc.roundtrip_ok(PROG)
    assert "def solve(values, limit)" in out  # entry point and its parameters untouched


def test_detect_reads_embedded_bits():
    d = mc.detect(mc.embed(PROG))
    assert d["n_votes"] > 0 and d["agree"] == d["n_votes"]  # every readable vote matches the codeword


def test_naming_conventions():
    for b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert mc.read_convention(mc.render(["max", "value"], b)) == b
    assert mc.split_words("x") is None and mc.split_words("var2x") is None
