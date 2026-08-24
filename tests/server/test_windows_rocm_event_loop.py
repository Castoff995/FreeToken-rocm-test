import asyncio

import uvicorn

from freetoken.server import loop as loop_compat


def test_windows_uvicorn_loop_supports_add_reader(monkeypatch):
    monkeypatch.setattr(loop_compat.sys, "platform", "win32")
    loop = loop_compat.uvicorn_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert callable(loop.add_reader)
    finally:
        loop.close()


def test_non_windows_uvicorn_loop_uses_platform_default(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(loop_compat.sys, "platform", "linux")
    monkeypatch.setattr(loop_compat.asyncio, "new_event_loop", lambda: sentinel)
    assert loop_compat.uvicorn_loop_factory() is sentinel


def test_uvicorn_accepts_repo_loop_factory():
    config = uvicorn.Config(lambda scope, receive, send: None, loop=loop_compat.uvicorn_loop_factory)
    assert config.get_loop_factory() is loop_compat.uvicorn_loop_factory
