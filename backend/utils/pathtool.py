"""Compatibility path helpers for the input-processing stack."""

from __future__ import annotations

from pathlib import Path

from backend.utils.path_tool import get_abs_path as _get_abs_path
from backend.utils.path_tool import get_root_path as _get_root_path


def get_root_path() -> Path:
    return Path(_get_root_path())


def get_abs_path(name: str) -> Path:
    return Path(_get_abs_path(name))
