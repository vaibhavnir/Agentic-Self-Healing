"""
Configuration loader for the self-healing pipeline.
Reads ``healing_strategies.yaml`` and returns typed dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "healing_strategies.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and return the full YAML configuration as a dict.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to ``config/healing_strategies.yaml``
        relative to the project root.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(
            "Config file not found at '%s'. Using empty configuration.", config_path
        )
        return {"strategies": [], "escalation": {}, "feedback": {}}

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    logger.info("Loaded config from '%s'", config_path)
    return data or {}
