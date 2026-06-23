"""Compatibility logger used by the migrated input-processing modules."""

from __future__ import annotations

import logging

from backend.utils.logger_handler import get_logger


class _CompatLogger:
    def __init__(self) -> None:
        self._logger = get_logger("input_processing")

    def debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        self._logger.exception(message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        self._logger.log(logging.INFO, message, *args, **kwargs)


log = _CompatLogger()
