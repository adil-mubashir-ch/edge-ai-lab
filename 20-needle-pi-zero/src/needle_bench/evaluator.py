from __future__ import annotations

from collections import Counter
from typing import Any


def normalise_value(value: Any) -> Any:
    """
    Recursively normalise values so that semantically identical
    JSON-like structures compare reliably.
    """
    if isinstance(value, float):
        return round(value, 8)

    if isinstance(value, dict):
        return {
            str(k): normalise_value(v)
            for k, v in sorted(value.items())
        }

    if isinstance(value, list):
        return [normalise_value(v) for v in value]

    return value


def normalise_calls(calls: Any) -> list[dict[str, Any]]:
    """
    Convert Needle function-call output into a consistent format.

    Supports both:
        {"name": "...", "arguments": {...}}

    and nested formats such as:
        {
            "function": {
                "name": "...",
                "arguments": {...}
            }
        }
    """
    if not calls:
        return []

    out: list[dict[str, Any]] = []

    for call in calls:
        if not isinstance(call, dict):
            continue

        function = call.get("function")

        if not isinstance(function, dict):
            function = {}

        name = call.get("name") or function.get("name")

        args = call.get("arguments")

        if args is None:
            args = function.get("arguments", {})

        if not isinstance(args, dict):
            args = {}

        out.append(
            {
                "name": name,
                "arguments": normalise_value(args),
            }
        )

    return out


def extract_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract function calls from a raw Needle response.
    """
    if not isinstance(response, dict):
        return []

    return normalise_calls(response.get("function_calls", []))


def tool_correct(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    """
    Check whether the predicted tool names match the expected tool names.

    Order is intentionally ignored because parallel tool calls are
    semantically order-independent.

    Counter is used instead of set so duplicate calls are still meaningful.
    """
    predicted_names = Counter(
        call.get("name") for call in predicted
    )

    expected_names = Counter(
        call.get("name") for call in expected
    )

    return predicted_names == expected_names


def arguments_correct(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    """
    Check whether arguments are correct for the corresponding tools.

    Calls are matched by tool name rather than list position, making
    parallel-call evaluation order-independent.

    Duplicate calls are supported through Counter-like matching.
    """
    if len(predicted) != len(expected):
        return False

    remaining = list(expected)

    for predicted_call in predicted:
        predicted_name = predicted_call.get("name")
        predicted_args = normalise_value(
            predicted_call.get("arguments", {})
        )

        match_index = None

        for index, expected_call in enumerate(remaining):
            if expected_call.get("name") != predicted_name:
                continue

            expected_args = normalise_value(
                expected_call.get("arguments", {})
            )

            if predicted_args == expected_args:
                match_index = index
                break

        if match_index is None:
            return False

        remaining.pop(match_index)

    return len(remaining) == 0


def exact_call_correct(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    """
    Headline metric.

    A prediction is exact when the complete set of tool calls and their
    arguments matches the expected calls.

    Parallel call ordering is ignored.
    """
    predicted_normalised = sorted(
        normalise_calls(predicted),
        key=lambda x: (
            x.get("name") or "",
            str(x.get("arguments", {})),
        ),
    )

    expected_normalised = sorted(
        normalise_calls(expected),
        key=lambda x: (
            x.get("name") or "",
            str(x.get("arguments", {})),
        ),
    )

    return predicted_normalised == expected_normalised


def call_set_metrics(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Identify extra and missing tool calls.

    Example:

        expected:  [control_lights]
        predicted: [control_lights, start_robot_vacuum]

    returns:

        extra  = ["start_robot_vacuum"]
        missing = []

    Duplicate calls are handled correctly.
    """
    predicted_names = Counter(
        call.get("name") for call in predicted
    )

    expected_names = Counter(
        call.get("name") for call in expected
    )

    extra = list(
        (predicted_names - expected_names).elements()
    )

    missing = list(
        (expected_names - predicted_names).elements()
    )

    return extra, missing


def evaluate_response(
    response: dict[str, Any],
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate one raw Needle response against the expected tool calls.

    This function deliberately evaluates the raw `complete()` response
    rather than executing any returned tools.
    """
    predicted = extract_calls(response)

    expected = normalise_calls(expected)

    extra_tools, missing_tools = call_set_metrics(
        predicted,
        expected,
    )

    return {
        # Core benchmark metrics
        "predicted_calls": predicted,
        "expected_calls": expected,
        "tool_correct": tool_correct(predicted, expected),
        "arguments_correct": arguments_correct(predicted, expected),
        "exact_call_correct": exact_call_correct(
            predicted,
            expected,
        ),

        # Diagnostic call-level metrics
        "extra_tools": extra_tools,
        "missing_tools": missing_tools,
        "num_predicted_calls": len(predicted),
        "num_expected_calls": len(expected),
        "num_extra_tools": len(extra_tools),
        "num_missing_tools": len(missing_tools),

        # Refusal / no-call metrics
        "is_refusal_expected": len(expected) == 0,
        "is_refusal_predicted": len(predicted) == 0,

        # Needle runtime metrics
        "confidence": response.get("confidence"),
        "prefill_tps": response.get("prefill_tps"),
        "decode_tps": response.get("decode_tps"),
        "peak_ram_mb_engine": response.get("peak_ram_mb"),

        # Needle response metadata
        "response_type": response.get("type"),
        "success": response.get("success"),
        "error": response.get("error"),
        "error_code": response.get("error_code"),
        "reasoning": response.get("reasoning"),
        "validation": response.get("validation"),
    }