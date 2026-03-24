"""
tradIA Live Trading — Structured Logging
Dual output: console (colored) + rotating daily log files.
API keys are automatically redacted from all log output.
"""
import logging
import logging.handlers
import re
import sys
from datetime import datetime
from pathlib import Path

from live.config_live import LOG_DIR


class _KeyRedactFilter(logging.Filter):
    """Redact API keys and secrets from log records."""
    _PATTERNS = [
        re.compile(r'(api[_\s]?key\s*[=:]\s*)\S+', re.IGNORECASE),
        re.compile(r'(api[_\s]?secret\s*[=:]\s*)\S+', re.IGNORECASE),
        re.compile(r'(password\s*[=:]\s*)\S+', re.IGNORECASE),
        re.compile(r'(secret\s*[=:]\s*)\S+', re.IGNORECASE),
    ]

    def filter(self, record):
        msg = record.getMessage()
        for p in self._PATTERNS:
            msg = p.sub(r'\1***REDACTED***', msg)
        record.msg = msg
        record.args = ()
        return True


def setup_logger(name: str = "tradIA") -> logging.Logger:
    """
    Configure and return the main application logger.

    - Console handler: INFO level, concise format
    - File handler: DEBUG level, detailed format, daily rotation (7-day keep)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.addFilter(_KeyRedactFilter())

    # ── Console handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # ── File handler (daily rotation) ──
    log_path = LOG_DIR / f"tradia_{datetime.utcnow():%Y-%m-%d}.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        interval=1,
        backupCount=30,        # keep 30 days
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


# Module-level convenience
log = setup_logger()
