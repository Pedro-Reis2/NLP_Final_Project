"""Post-generation multi-channel code watermark: our reimplementation of Choi, Gao & Duan (2026),
"Multi-Channel Spread-Spectrum Code Watermarking" (arXiv:2607.06009).

The authors' repository (github.com/soohyeonc/Multi_Channel_Watermarking, linked in the paper) was
not public when we ran this study, so this follows the paper's description. Differences:
  - Structural channels: 5 of the paper's 8 pairs (aug_assign, compare_dir, equality_dir,
    empty_coll, none_check). list_comp and ternary are omitted (complex rewrites); operand_order is
    omitted because a + b -> b + a is not behaviour-preserving for strings/lists.
  - Naming channel: only names that are already multi-word (the paper also extends single-word
    names, e.g. n -> n_count). Parameters of top-level functions are never renamed (callers).
  - Key -> permutation: one Fisher-Yates shuffle of the B codeword positions per cycle, seeded by
    SHA-256(key | cycle) (the paper derives the seeds from secp256k1 multiples of the key).
  - Code: the paper's "compact" systematic binary linear code, 24 message bits + 4t parity bits
    (t = 2 -> B = 32) with a parity matrix from a fixed SHA-256 seed; decoding solves the erased
    bits from the parity checks and fails on any inconsistency or ambiguity.
  - The code is regenerated with ast.unparse, so `#` comments are not preserved.

Embedding: canonicalise every site (snake_case names, bit-0 structural forms), order the sites by
a walk of the canonical tree (so the order does not depend on the bits embedded), assign site k to
codeword position phi(k), and set each site to the form of its bit. Detection: canonicalise the
suspect code the same way, read the bit each site carried, vote per position (majority; ties =
erasure), decode. Two detectors:
  exact  - the paper's rule: decoding recovers exactly the 24-bit payload
  z      - vote agreement: z = (agree - n/2) / sqrt(n/4) over the n readable votes, compared
           with the expected codeword; thresholded empirically at 1% FPR like the other methods
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import keyword
import math
import random
import re

PAYLOAD = "qwen"  # 4 chars x 6 bits = 24 bits (the paper's model-ID payload)
KEY = "15485863"  # same secret as the decoding-time watermarks (MarkLLM default hash key)
T = 2
N_MSG, N_PAR = 24, 4 * T
B = N_MSG + N_PAR

_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


class NotEmbeddable(Exception):
    pass


# ------------------------------------------------------------------ payload code
def payload_bits(s: str = PAYLOAD) -> list[int]:
    out = []
    for ch in s.ljust(4, "\0")[:4]:
        v = _ALPHABET.index(ch) if ch in _ALPHABET else 63
        out += [(v >> (5 - i)) & 1 for i in range(6)]
    return out


def _parity_matrix() -> list[list[int]]:
    rng = random.Random(int(hashlib.sha256(b"mcw-parity-v1").hexdigest(), 16))
    return [[rng.randint(0, 1) for _ in range(N_MSG)] for _ in range(N_PAR)]


P = _parity_matrix()


def encode(msg: list[int]) -> list[int]:
    return msg + [sum(p * m for p, m in zip(row, msg)) % 2 for row in P]


CODEWORD = encode(payload_bits())


def decode(bits: list[int | None]) -> list[int] | None:
    """bits: 0/1 or None (erasure). Solve the erased bits from H c = 0 (H = [P | I]) by Gaussian
    elimination over GF(2). Returns the 24 message bits, or None if inconsistent/ambiguous."""
    H = [row + [1 if i == j else 0 for j in range(N_PAR)] for i, row in enumerate(P)]
    unk = [j for j in range(B) if bits[j] is None]
    rows = []
    for h in H:
        rhs = sum(h[j] * bits[j] for j in range(B) if bits[j] is not None) % 2
        rows.append([h[j] for j in unk] + [rhs])
    n, r = len(unk), 0
    for col in range(n):
        piv = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if piv is None:
            return None  # an erased bit not determined by the checks
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        r += 1
    if any(row[-1] for row in rows[r:]):
        return None  # parity checks violated: errors present
    full = list(bits)
    for k, j in enumerate(unk):
        full[j] = rows[k][-1]
    return full[:N_MSG]


def phi(k: int) -> int:
    """Codeword position of slot k: a keyed permutation of the B positions per full cycle."""
    cyc, off = divmod(k, B)
    perm = list(range(B))
    random.Random(int(hashlib.sha256(f"{KEY}|{cyc}".encode()).hexdigest(), 16)).shuffle(perm)
    return perm[off]


# ------------------------------------------------------------------ naming channel
_RESERVED = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def split_words(name: str) -> list[str] | None:
    core = name.strip("_")
    if core != name or not core or not core.isascii():
        return None  # leading/trailing underscores: leave alone
    if "_" in core:
        parts = core.split("_")
        if any(not p for p in parts):
            return None
        if not (core == core.lower() or core == core.upper()):
            return None
        words = [p.lower() for p in parts]
    else:
        words = [w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", core)]
        if "".join(words) != core.lower():
            return None
    if len(words) < 2 or not all(w[0].isalpha() for w in words):
        return None
    return words


def read_convention(name: str) -> tuple[int, int] | None:
    w = split_words(name)
    if w is None:
        return None
    if "_" in name:
        return (0, 0) if name == name.lower() else (1, 1) if name == name.upper() else None
    if name[0].islower():
        return (0, 1)
    return (1, 0) if name[0].isupper() and name != name.upper() else None


def render(words: list[str], b: tuple[int, int]) -> str:
    if b == (0, 0):
        return "_".join(words)
    if b == (0, 1):
        return words[0] + "".join(w.capitalize() for w in words[1:])
    if b == (1, 0):
        return "".join(w.capitalize() for w in words)
    return "_".join(w.upper() for w in words)


def _blocked_names(tree: ast.AST) -> set[str]:
    """Names that must not be renamed: function/class names, parameters of top-level functions,
    attributes, keyword-argument names, imports, globals/nonlocals."""
    blocked = set()
    top_funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] \
        if isinstance(tree, ast.Module) else []
    for f in top_funcs:
        a = f.args
        blocked |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        blocked |= {x.arg for x in (a.vararg, a.kwarg) if x}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            blocked.add(n.name)
        elif isinstance(n, ast.Attribute):
            blocked.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            blocked.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            blocked |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            blocked |= set(n.names)
    return blocked


def _variables(tree: ast.AST) -> list[str]:
    """Eligible multi-word variables, in order of first appearance (ast.walk order)."""
    blocked = _blocked_names(tree)
    stored, order = set(), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            stored.add(n.id)
        elif isinstance(n, ast.arg):
            stored.add(n.arg)
    for n in ast.walk(tree):
        nm = n.id if isinstance(n, ast.Name) else n.arg if isinstance(n, ast.arg) else None
        if nm and nm in stored and nm not in blocked and nm not in _RESERVED and nm not in order \
                and split_words(nm):
            order.append(nm)
    # two names that normalise to the same words would collide: drop both
    key = {}
    for nm in order:
        key.setdefault(tuple(split_words(nm)), []).append(nm)
    return [nm for nm in order if len(key[tuple(split_words(nm))]) == 1]


def _rename(tree: ast.AST, mapping: dict[str, str]) -> None:
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in mapping:
            n.id = mapping[n.id]
        elif isinstance(n, ast.arg) and n.arg in mapping:
            n.arg = mapping[n.arg]


# ------------------------------------------------------------------ structural channels
_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
          ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift)
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}


def _is_none(n):
    return isinstance(n, ast.Constant) and n.value is None


class _Canon(ast.NodeTransformer):
    """Post-order: canonicalise children, then this node. Marks every structural site with
    _wm_kind and the bit it carried before canonicalisation (_wm_bit)."""

    def _mark(self, node, kind, bit):
        node._wm_kind, node._wm_bit = kind, bit
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and isinstance(node.op, _ARITH):
            new = ast.Assign(targets=[ast.Name(node.target.id, ast.Store())],
                             value=ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, node.value))
            return self._mark(ast.copy_location(new, node), "aug_assign", 1)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        t, v = node.targets, node.value
        if (len(t) == 1 and isinstance(t[0], ast.Name) and isinstance(v, ast.BinOp)
                and isinstance(v.left, ast.Name) and v.left.id == t[0].id and isinstance(v.op, _ARITH)):
            return self._mark(node, "aug_assign", 0)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        op, l, r = node.ops[0], node.left, node.comparators[0]
        if type(op) in (ast.Lt, ast.LtE):
            return self._mark(node, "compare_dir", 0)
        if type(op) in (ast.Gt, ast.GtE):
            new = ast.Compare(left=r, ops=[_FLIP[type(op)]()], comparators=[l])
            return self._mark(ast.copy_location(new, node), "compare_dir", 1)
        if type(op) in (ast.Is, ast.IsNot) and _is_none(r):
            new = ast.Compare(left=l, ops=[ast.Eq() if isinstance(op, ast.Is) else ast.NotEq()], comparators=[r])
            return self._mark(ast.copy_location(new, node), "none_check", 1)
        if type(op) in (ast.Eq, ast.NotEq):
            if _is_none(r) and not _is_none(l):
                return self._mark(node, "none_check", 0)
            if _is_none(l) or _is_none(r):
                return node
            kl, kr = ast.dump(l), ast.dump(r)
            if kl == kr:
                return node
            if kl < kr:
                return self._mark(node, "equality_dir", 0)
            new = ast.Compare(left=r, ops=[op], comparators=[l])
            return self._mark(ast.copy_location(new, node), "equality_dir", 1)
        return node

    def visit_List(self, node):
        self.generic_visit(node)
        if not node.elts and isinstance(node.ctx, ast.Load):
            return self._mark(node, "empty_coll", 0)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        return self._mark(node, "empty_coll", 0) if not node.keys else node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id in ("list", "dict")
                and not node.args and not node.keywords):
            new = ast.List(elts=[], ctx=ast.Load()) if node.func.id == "list" else ast.Dict(keys=[], values=[])
            return self._mark(ast.copy_location(new, node), "empty_coll", 1)
        return node


def _to_alt(node):
    """Bit-1 form of a canonical site (children reused)."""
    k = node._wm_kind
    if k == "aug_assign":
        return ast.AugAssign(target=ast.Name(node.targets[0].id, ast.Store()), op=node.value.op,
                             value=node.value.right)
    if k == "compare_dir":
        return ast.Compare(left=node.comparators[0], ops=[_FLIP[type(node.ops[0])]()], comparators=[node.left])
    if k == "equality_dir":
        return ast.Compare(left=node.comparators[0], ops=node.ops, comparators=[node.left])
    if k == "none_check":
        return ast.Compare(left=node.left, ops=[ast.Is() if isinstance(node.ops[0], ast.Eq) else ast.IsNot()],
                           comparators=node.comparators)
    if k == "empty_coll":
        return ast.Call(func=ast.Name("list" if isinstance(node, ast.List) else "dict", ast.Load()),
                        args=[], keywords=[])
    raise ValueError(k)


class _SetBits(ast.NodeTransformer):
    def __init__(self, bit_of):
        self.bit_of = bit_of  # id(node) -> target bit

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if id(node) in self.bit_of and self.bit_of[id(node)] == 1:
            return ast.copy_location(_to_alt(node), node)
        return node

    def visit(self, node):
        return self.generic_visit(node)


# ------------------------------------------------------------------ pool
def _canonical(src: str):
    """Parse, canonicalise names and structure. Returns (tree, slots) where slots is the ordered
    list of (kind, ref, bit_read): kind "name" (2 slots, ref = canonical snake name) or a structural
    kind (1 slot, ref = canonical node)."""
    tree = ast.parse(src)
    variables = _variables(tree)
    name_bits = {v: read_convention(v) for v in variables}
    snake = {v: render(split_words(v), (0, 0)) for v in variables}
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | _blocked_names(tree)
    if any(s in taken and s != v for v, s in snake.items()):
        clash = {v for v, s in snake.items() if s in taken and s != v}
        variables = [v for v in variables if v not in clash]
    _rename(tree, {v: snake[v] for v in variables})
    tree = _Canon().visit(tree)
    ast.fix_missing_locations(tree)
    slots = [("name", snake[v], name_bits[v]) for v in variables]
    slots += [(n._wm_kind, n, n._wm_bit) for n in ast.walk(tree) if hasattr(n, "_wm_kind")]
    return tree, slots


def _slot_positions(slots):
    """Codeword position(s) of every slot, in order (a name takes two consecutive slots)."""
    pos, k = [], 0
    for kind, _, _ in slots:
        w = 2 if kind == "name" else 1
        pos.append([phi(k + i) for i in range(w)])
        k += w
    return pos


def embed(src: str) -> str:
    """Watermark `src` with CODEWORD. Raises NotEmbeddable if it has no sites."""
    tree, slots = _canonical(src)
    if not slots:
        raise NotEmbeddable("no sites")
    pos = _slot_positions(slots)
    rename, bit_of = {}, {}
    for (kind, ref, _), ps in zip(slots, pos):
        if kind == "name":
            rename[ref] = render(ref.split("_"), (CODEWORD[ps[0]], CODEWORD[ps[1]]))
        else:
            bit_of[id(ref)] = CODEWORD[ps[0]]
    _rename(tree, rename)
    tree = _SetBits(bit_of).visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree) + "\n"
    ast.parse(out)
    return out


def detect(src: str) -> dict:
    """Votes, agreement z-score and exact decoding for `src`."""
    try:
        _, slots = _canonical(src)
    except (SyntaxError, ValueError, RecursionError):
        return {"n_sites": 0, "n_votes": 0, "agree": 0, "z": None, "exact": False, "n_erased": B}
    pos = _slot_positions(slots)
    v0, v1 = [0] * B, [0] * B
    agree = n = 0
    for (kind, _, bit), ps in zip(slots, pos):
        bits = bit if kind == "name" else (bit,)
        if bits is None:
            continue
        for b, p in zip(bits, ps):
            (v1 if b else v0)[p] += 1
            n += 1
            agree += int(b == CODEWORD[p])
    word = [0 if a > b else 1 if b > a else None for a, b in zip(v0, v1)]
    msg = decode(word)
    return {"n_sites": len(slots), "n_votes": n, "agree": agree,
            "z": (agree - n / 2) / math.sqrt(n / 4) if n else None,
            "exact": msg == payload_bits(), "n_erased": sum(w is None for w in word)}


def roundtrip_ok(src: str) -> bool:
    """Sanity: canonical form of the watermarked code equals that of the original."""
    a = ast.dump(_canonical(src)[0])
    b = ast.dump(_canonical(embed(src))[0])
    return a == b
