"""Resumable JSONL I/O and run manifests."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from .config import LOGS, OUT


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a crash may leave a truncated last line; ignore it
                pass
    return rows


def done_ids(path: Path, key: str = "id") -> set:
    return {r[key] for r in read_jsonl(path) if key in r}


def append_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except Exception:
        return "n/a"


def write_manifest(name: str, payload: dict) -> Path:
    path = OUT / "manifests" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.update(
        git_sha=git_sha(),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version.split()[0],
        versions={p: _ver(p) for p in
                  ["torch", "transformers", "markllm", "evalplus", "black"]},
        argv=sys.argv,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def get_logger(step: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(step)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(LOGS / f"{step}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log
