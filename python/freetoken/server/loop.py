from __future__ import annotations

import asyncio
import sys


def uvicorn_loop_factory() -> asyncio.AbstractEventLoop:
    """Return a pyzmq-compatible loop on Windows and the platform default elsewhere."""

    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
