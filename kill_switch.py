"""
Process-wide kill switch + a small FastAPI service that toggles it.

The flag is a thread-safe ``Event`` (so the bot's evaluation loop —
running in the main thread / event loop — can read it freely while the
HTTP server thread mutates it).

Auth: bearer token from ``.env`` (``KILL_SWITCH_TOKEN``).  Wrong / missing
token returns 401.

Endpoints
---------
GET    /status   → ``{active, since}``
POST   /kill     → activate; bot will flatten on next tick
POST   /reset    → deactivate

The bot wires it like so:
    switch = KillSwitch(token=...)
    exit_mgr = ExitManager(kill_switch_active=lambda: switch.active)
    # then run `uvicorn run` in the background:
    serve_kill_switch(switch, port=8085)
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException, status


class KillSwitch:
    def __init__(self, token: str) -> None:
        if not token or len(token) < 16:
            raise ValueError(
                "KILL_SWITCH_TOKEN must be set to a non-trivial value (≥16 chars). "
                "Refusing to start with a weak token."
            )
        self._token = token
        self._flag = threading.Event()
        self._activated_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self._flag.is_set()

    @property
    def activated_at(self) -> datetime | None:
        return self._activated_at

    def activate(self) -> None:
        self._flag.set()
        self._activated_at = datetime.now(tz=timezone.utc)

    def reset(self) -> None:
        self._flag.clear()
        self._activated_at = None

    def _check_token(self, authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or malformed Authorization header")
        supplied = authorization[len("Bearer "):].strip()
        if supplied != self._token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def build_app(switch: KillSwitch) -> FastAPI:
    app = FastAPI(title="Trading Bot Kill Switch", version="1.0")

    @app.get("/status")
    def status_endpoint(authorization: Annotated[str | None, Header()] = None) -> dict:
        switch._check_token(authorization)
        return {
            "active": switch.active,
            "since": switch.activated_at.isoformat() if switch.activated_at else None,
        }

    @app.post("/kill")
    def kill(authorization: Annotated[str | None, Header()] = None) -> dict:
        switch._check_token(authorization)
        switch.activate()
        return {"active": True, "since": switch.activated_at.isoformat()}

    @app.post("/reset")
    def reset_endpoint(authorization: Annotated[str | None, Header()] = None) -> dict:
        switch._check_token(authorization)
        switch.reset()
        return {"active": False}

    return app


def serve_kill_switch(switch: KillSwitch, *, host: str = "127.0.0.1", port: int = 8085) -> None:  # pragma: no cover
    """Blocking — call from a dedicated thread / process."""
    app = build_app(switch)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
