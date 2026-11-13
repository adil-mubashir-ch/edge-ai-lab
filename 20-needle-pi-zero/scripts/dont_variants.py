"""How many ways can you say "don't" before Needle does it anyway?

Same setup as dont_demo.py (cactus run(), fake tools that print instead of switching anything), but
with many phrasings of the same forbidden action, at 5 and 50 tools. Needle is deterministic, so one
run per prompt is enough. Each prompt ends up as one of:
    DID IT    the forbidden action was executed
    opposite  it executed the opposite action instead (lights off, fan off)
    blocked   nothing was executed
    other     something else was executed

    NEEDLE_WEIGHTS=needle3.cact PYTHONPATH=src python scripts/dont_variants.py
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus

from needle_bench.tools import tools_for_count

LIGHTS = ("control_lights", {"room": "bedroom", "action": "on"})
FAN = ("control_fan", {"room": "study", "action": "on"})

# (prompt, forbidden call). The forbidden call is matched on name plus these arguments.
PROMPTS = [
    ("Do not turn on the bedroom lights.", LIGHTS),
    ("Don't turn on the bedroom lights.", LIGHTS),
    ("Do NOT turn on the bedroom lights.", LIGHTS),
    ("Please don't turn on the bedroom lights.", LIGHTS),
    ("Never turn on the bedroom lights.", LIGHTS),
    ("Don't switch on the bedroom lights.", LIGHTS),
    ("No, don't turn on the bedroom lights.", LIGHTS),
    ("I don't want the bedroom lights on.", LIGHTS),
    ("Avoid turning on the bedroom lights.", LIGHTS),
    ("Leave the bedroom lights off.", LIGHTS),
    ("Keep the bedroom lights off.", LIGHTS),
    ("The bedroom lights should stay off.", LIGHTS),
    ("Turn on the bedroom lights. Actually, no, don't.", LIGHTS),
    ("Don't start the fan in the study.", FAN),
    ("Do not start the fan in the study.", FAN),
    ("Don't turn on the study fan.", FAN),
    ("Never turn on the fan in the study.", FAN),
    ("Please don't start the fan in the study.", FAN),
    # Controls: plain requests that should go through.
    ("Turn on the bedroom lights.", None),
    ("Start the fan in the study.", None),
]


def matches(call, forbidden):
    name, args = forbidden
    return call["name"] == name and all(call["arguments"].get(k) == v for k, v in args.items())


def opposite(forbidden):
    name, args = forbidden
    return name, {**args, "action": "off"}


def main():
    weights = os.environ.get("NEEDLE_WEIGHTS", "needle3.cact")
    print(f"weights: {weights}", flush=True)
    for count in (5, 50):
        executed = []

        def fake(name):
            def fn(**arguments):
                executed.append({"name": name, "arguments": arguments})
                return {"ok": True}
            return fn

        tools = []
        for schema in tools_for_count(count):
            fn = fake(schema["name"])
            fn._cactus_tool = schema
            tools.append(fn)
        agent = cactus.Cactus(tools=tools, weights=weights, stateless=True)

        print(f"\n=== {count} tools", flush=True)
        for prompt, forbidden in PROMPTS:
            executed.clear()
            start = time.perf_counter()
            response = agent.run(prompt, max_steps=1)
            seconds = time.perf_counter() - start
            if forbidden is None:
                verdict = "control"
            elif any(matches(c, forbidden) for c in executed):
                verdict = "DID IT"
            elif not executed:
                verdict = "blocked"
            elif all(matches(c, opposite(forbidden)) for c in executed):
                verdict = "opposite"
            else:
                verdict = "other"
            record = {
                "tools": count,
                "prompt": prompt,
                "verdict": verdict,
                "executed": executed[:],
                "validation": response.get("validation"),
                "seconds": round(seconds, 2),
            }
            print(f"{verdict:9} {prompt!r:55} {json.dumps(executed)}", flush=True)
            print("JSON " + json.dumps(record), flush=True)
        agent.close()


if __name__ == "__main__":
    main()
