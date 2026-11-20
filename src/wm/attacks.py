"""Semantics-preserving code attacks.

All attacks are source-to-source and edit the original text in place where
possible (no ast.unparse), so each attack changes only what it is meant to
change. Every attack returns the new source or raises AttackError.

Attacks (ids used in output files):
  strip_comments   remove # comments and docstrings
  black            reformat with black (default mode)
  rename_1/2/all   rename 1, 2 or all eligible local identifiers
  deadcode_try     insert a dead `if False:` block and wrap function bodies in
                   try/except-reraise
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import keyword
import tokenize

ATTACKS = ["strip_comments", "black", "rename_1", "rename_2", "rename_all", "deadcode_try"]


class AttackError(Exception):
    pass


# ----------------------------------------------------------------- helpers
def _lines(src: str) -> list[str]:
    return src.splitlines(keepends=True)


def _parse(src: str) -> ast.Module:
    try:
        return ast.parse(src)
    except SyntaxError as e:  # attacks only run on parseable code
        raise AttackError(f"unparseable input: {e}") from e


def _byte_to_char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


# ----------------------------------------------------------- strip_comments
def _docstring_nodes(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                yield node, body[0]


def strip_comments(src: str) -> str:
    _parse(src)
    lines = _lines(src)
    # 1) comments, via tokenize positions
    comment_at: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comment_at[tok.start[0]] = tok.start[1]
    out = []
    for i, line in enumerate(lines, start=1):
        if i in comment_at:
            nl = "\n" if line.endswith("\n") else ""
            kept = line[: comment_at[i]].rstrip()
            if kept.strip() == "":
                continue  # comment-only line: drop it
            line = kept + nl
        out.append(line)
    src2 = "".join(out)

    # 2) docstrings, via AST line ranges (bottom-up so line numbers stay valid)
    tree = _parse(src2)
    lines = _lines(src2)
    docs = sorted(_docstring_nodes(tree), key=lambda p: -p[1].lineno)
    for parent, doc in docs:
        start, end = doc.lineno, doc.end_lineno
        if not isinstance(parent, ast.Module) and parent.body[0].lineno == parent.lineno:
            continue  # `def f(): """doc"""` on one line: leave it
        first = lines[start - 1]
        indent = first[: len(first) - len(first.lstrip())]
        repl = [indent + "pass\n"] if len(parent.body) == 1 else []
        lines[start - 1:end] = repl
    out = "".join(lines)
    _parse(out)
    return out


# -------------------------------------------------------------------- black
def black_format(src: str) -> str:
    import black
    try:
        return black.format_str(src, mode=black.Mode())
    except Exception as e:
        raise AttackError(f"black failed: {e}") from e


# ------------------------------------------------------------------- rename
_BUILTINS = set(dir(builtins))


def _rename_candidates(tree: ast.Module) -> list[str]:
    """Local identifiers that can be renamed consistently file-wide.

    Excluded: function/class names, imported names, builtins, keywords,
    dunders, `self`/`cls`, global/nonlocal names, names assigned in class
    bodies (they become attributes), names used as keyword arguments in calls
    (renaming a parameter would break `f(x=...)`), exception-handler names.
    """
    excluded = set(_BUILTINS) | set(keyword.kwlist) | {"self", "cls"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            excluded.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                excluded.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            excluded.update(node.names)
        elif isinstance(node, ast.keyword) and node.arg:
            excluded.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            excluded.add(node.name)
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        excluded.add(sub.id)
    # first-occurrence order of bound names (params, assignments, loop targets)
    seen: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        name = pos = None
        if isinstance(node, ast.arg):
            name, pos = node.arg, (node.lineno, node.col_offset)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name, pos = node.id, (node.lineno, node.col_offset)
        if name and name not in excluded and not name.startswith("__"):
            if name not in seen or pos < seen[name]:
                seen[name] = pos
    return sorted(seen, key=lambda n: seen[n])


def rename(src: str, k: int | None) -> str:
    """Rename the first k eligible identifiers (k=None → all)."""
    tree = _parse(src)
    cands = _rename_candidates(tree)
    if not cands:
        raise AttackError("no renameable identifiers")
    targets = cands if k is None else cands[:k]
    all_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    mapping, i = {}, 0
    for t in targets:
        while f"var_{i}" in all_names:
            i += 1
        mapping[t] = f"var_{i}"
        i += 1
    # collect exact positions (line, byte col) of every occurrence
    edits = []  # (line, char_col, old, new)
    lines = _lines(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            old, ln, bc = node.id, node.lineno, node.col_offset
        elif isinstance(node, ast.arg) and node.arg in mapping:
            old, ln, bc = node.arg, node.lineno, node.col_offset
        else:
            continue
        cc = _byte_to_char_col(lines[ln - 1], bc)
        if lines[ln - 1][cc: cc + len(old)] != old:
            raise AttackError(f"position mismatch for {old} at {ln}:{cc}")
        edits.append((ln, cc, old, mapping[old]))
    for ln, cc, old, new in sorted(set(edits), key=lambda e: (-e[0], -e[1])):
        line = lines[ln - 1]
        lines[ln - 1] = line[:cc] + new + line[cc + len(old):]
    out = "".join(lines)
    _parse(out)
    return out


# ------------------------------------------------------------- deadcode_try
_DEAD_TEMPLATES = [
    ["if False:", "    _unused = 0", "    print(_unused)"],
    ["if 0:", "    _tmp = []", "    _tmp.append(1)"],
    ["while False:", "    _x = None", "    break"],
]


def _multiline_string_rows(src: str) -> list[tuple[int, int]]:
    spans = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            spans.append((tok.start[0], tok.end[0]))
    fstart = getattr(tokenize, "FSTRING_START", None)
    if fstart is not None:  # py3.12 f-strings
        stack = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == fstart:
                stack.append(tok.start[0])
            elif tok.type == tokenize.FSTRING_END and stack:
                s = stack.pop()
                if s != tok.end[0]:
                    spans.append((s, tok.end[0]))
    return spans


def deadcode_try(src: str, salt: str = "") -> str:
    tree = _parse(src)
    lines = _lines(src)
    mstr = _multiline_string_rows(src)
    funcs = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            funcs += [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        raise AttackError("no function to wrap")
    h = int(hashlib.md5((salt + src).encode()).hexdigest(), 16)
    dead = _DEAD_TEMPLATES[h % len(_DEAD_TEMPLATES)]
    changed = False
    for fn in sorted(funcs, key=lambda f: -f.lineno):
        body = fn.body
        if body[0].lineno == fn.lineno:
            continue  # one-line def
        has_doc = (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                   and isinstance(body[0].value.value, str))
        stmts = body[1:] if has_doc else body
        first_line = body[0].lineno
        indent_src = lines[first_line - 1]
        ind = indent_src[: len(indent_src) - len(indent_src.lstrip())]
        dead_block = [ind + d + "\n" for d in dead]
        if not stmts:
            insert_at = body[0].end_lineno  # after docstring
            lines[insert_at:insert_at] = dead_block
            changed = True
            continue
        start, end = stmts[0].lineno, fn.end_lineno
        # trailing comment lines after the last stmt belong to the function too
        end = max(end, stmts[-1].end_lineno)
        wrap = not any(s <= end and e >= start for s, e in mstr)
        if wrap:
            inner = [("    " + l) if l.strip() else l for l in lines[start - 1:end]]
            if inner and not inner[-1].endswith("\n"):
                inner[-1] += "\n"
            block = (dead_block + [ind + "try:\n"] + inner
                     + [ind + "except Exception:\n", ind + "    raise\n"])
            lines[start - 1:end] = block
        else:
            lines[start - 1:start - 1] = dead_block
        changed = True
    if not changed:
        raise AttackError("nothing changed")
    out = "".join(lines)
    _parse(out)
    return out


# ----------------------------------------------------------------- dispatch
def apply_attack(name: str, src: str, salt: str = "") -> str:
    if name == "strip_comments":
        return strip_comments(src)
    if name == "black":
        return black_format(src)
    if name == "rename_1":
        return rename(src, 1)
    if name == "rename_2":
        return rename(src, 2)
    if name == "rename_all":
        return rename(src, None)
    if name == "deadcode_try":
        return deadcode_try(src, salt)
    raise ValueError(name)
