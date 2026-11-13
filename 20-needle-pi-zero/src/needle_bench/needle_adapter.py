from __future__ import annotations
import hashlib
import os
from typing import Any

# Needle 3 ships in cactus-compute (`import cactus`) since 5 Nov 2026; Needle 2
# stays on the old cactus-needle package (`import needle`).
def _runtime():
    try:
        import cactus
        return cactus, cactus.Cactus
    except ImportError:
        import needle
        return needle, needle.Needle

def needle_version() -> str:
    try:
        from importlib.metadata import version
        module, _ = _runtime()
        dist = "cactus-compute" if module.__name__ == "cactus" else "cactus-needle"
        return f"{dist} {version(dist)}"
    except Exception:
        return "not-importable"

def weights_path() -> str:
    return os.environ.get(
        "NEEDLE_WEIGHTS",
        os.path.join(os.getcwd(), "needle2.cact")
    )

def weights_info() -> dict[str, Any]:
    path = weights_path()
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
    }

def build_agent(tools: list[dict[str, Any]], system: str | None = None):
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("CACTUS_TELEMETRY", "0")
    _, agent_class = _runtime()

    weights = weights_path()

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Needle weights not found: {weights}"
        )

    # The runtime reads the generation (2 or 3) from the weights file.
    return agent_class(
        tools=tools,
        system=system,
        weights=weights,
    )

def complete(agent, prompt: str, max_new_tokens: int = 256) -> dict[str, Any]:
    response = agent.complete(prompt, max_new_tokens=max_new_tokens)
    if not isinstance(response, dict):
        raise TypeError(f"Needle returned {type(response)!r}, expected dict")
    return response
