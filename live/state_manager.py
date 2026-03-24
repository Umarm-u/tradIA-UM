"""
tradIA Live Trading — State Manager
Persists bot state to disk for crash recovery.
Uses atomic writes to prevent corruption.
"""
import json
import os
import tempfile
import time
from typing import Optional, Dict

from live.config_live import STATE_FILE
from live.logger import log


class StateManager:
    """
    JSON-based state persistence for the trading bot.
    Saves position details, daily PnL, trailing stop state,
    and the last processed candle timestamp.
    """

    def __init__(self, state_file: str = None):
        self.state_file = str(state_file or STATE_FILE)

    def save(
        self,
        position: Optional[Dict] = None,
        daily_pnl: float = 0.0,
        daily_trades: int = 0,
        last_candle_time: str = "",
        trailing_state: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ):
        """
        Save bot state to disk atomically.

        Atomic write: write to temp file, then rename (os.replace).
        This prevents corruption if the process crashes mid-write.
        """
        state = {
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "position": position,
            "daily_pnl": daily_pnl,
            "daily_trades": daily_trades,
            "last_candle_time": last_candle_time,
            "trailing_state": trailing_state,
        }

        if extra:
            state.update(extra)

        try:
            # Write to temp file first
            dir_name = os.path.dirname(self.state_file) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, default=str)

            # Atomic rename
            os.replace(tmp_path, self.state_file)
            log.debug(f"State saved to {self.state_file}")

        except Exception as e:
            log.error(f"Failed to save state: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load(self) -> Optional[Dict]:
        """
        Load bot state from disk.
        Returns None if no state file exists or if it's corrupted.
        """
        if not os.path.exists(self.state_file):
            log.info("No saved state found — starting fresh")
            return None

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            log.info(
                f"State loaded from {self.state_file} "
                f"(saved at {state.get('timestamp_human', 'unknown')})"
            )
            return state

        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load state (corrupted?): {e}")
            return None

    def clear(self):
        """Remove saved state file."""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                log.info("State file cleared")
        except Exception as e:
            log.warning(f"Failed to clear state: {e}")

    def has_saved_position(self) -> bool:
        """Check if there's a saved position in state."""
        state = self.load()
        return state is not None and state.get("position") is not None
