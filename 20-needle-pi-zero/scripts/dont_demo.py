"""Does Needle actually *do* what it was told not to?

The benchmark uses Needle's complete(), which only returns the call it would make. This script uses
run(), the path a real app takes: it executes the calls. The tools are the benchmark's own
definitions (5-tool catalogue), attached to fake functions that print instead of switching anything.

    NEEDLE_WEIGHTS=needle3.cact PYTHONPATH=src python scripts/dont_demo.py
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus

from needle_bench.tools import tools_for_count

PROMPTS = [
    "Do not turn on the bedroom lights.",
    "Don't start the fan in the study.",
    "Turn on the bedroom lights.",
]


def fake(name):
    def fn(**arguments):
        print(f"    >>> EXECUTED {name}({json.dumps(arguments)})", flush=True)
        return {"ok": True}
    return fn


def main():
    weights = os.environ.get("NEEDLE_WEIGHTS", "needle3.cact")
    tools = []
    for schema in tools_for_count(5):
        fn = fake(schema["name"])
        fn._cactus_tool = schema
        tools.append(fn)

    agent = cactus.Cactus(tools=tools, weights=weights, stateless=True)
    print(f"weights: {weights}")
    for prompt in PROMPTS:
        print(f'\n"{prompt}"', flush=True)
        start = time.perf_counter()
        response = agent.run(prompt, max_steps=1)
        seconds = time.perf_counter() - start
        print(f"    results: {json.dumps(response.get('results'))}")
        print(f"    validation: {json.dumps(response.get('validation'))}  ({seconds:.1f} s)")
    agent.close()


if __name__ == "__main__":
    main()
