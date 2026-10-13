"""Tiny config loader: YAML file plus dotted command-line overrides.

    python -m src.evaluate --config configs/default.yaml \
        --model.backend hf --model.name Qwen/Qwen2.5-1.5B-Instruct \
        --task.family fsm --io.run_name fsm_qwen15
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(text: str) -> Any:
    """Turn a command-line string into the most specific type it parses as.

    JSON does the work: it handles numbers, lists, ``true``/``false`` and
    ``null``. Deliberately no extra aliases. An earlier version also mapped the
    bare word ``none`` to ``None``, which silently broke
    ``--intervention.kind none`` and ``--io.run_name none``: both are valid
    strings here. Write ``null`` if you really mean a null.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-mapping at '{part}'")
    node[parts[-1]] = value


def get_dotted(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse --config plus arbitrary --dotted.key value overrides."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default="configs/default.yaml")
    known, unknown = parser.parse_known_args(argv)

    cfg = copy.deepcopy(load_yaml(known.config))

    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise SystemExit(f"unexpected argument: {token}")
        if "=" in token:
            key, raw = token[2:].split("=", 1)
            i += 1
        else:
            key = token[2:]
            if i + 1 >= len(unknown):
                raise SystemExit(f"missing value for --{key}")
            raw = unknown[i + 1]
            i += 2
        set_dotted(cfg, key, _coerce(raw))

    cfg["_config_path"] = str(known.config)
    return cfg
