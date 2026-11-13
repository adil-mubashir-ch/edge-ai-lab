from needle_bench.evaluator import evaluate_response


def test_exact_positive_call():
    response = {
        "function_calls": [
            {
                "name": "control_lights",
                "arguments": {
                    "room": "study",
                    "action": "dim",
                    "brightness_percent": 30,
                },
            }
        ]
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "study",
                "action": "dim",
                "brightness_percent": 30,
            },
        }
    ]

    result = evaluate_response(response, expected)

    assert result["tool_correct"]
    assert result["arguments_correct"]
    assert result["exact_call_correct"]

    assert result["extra_tools"] == []
    assert result["missing_tools"] == []


def test_refusal():
    result = evaluate_response(
        {"function_calls": []},
        [],
    )

    assert result["tool_correct"]
    assert result["arguments_correct"]
    assert result["exact_call_correct"]

    assert result["is_refusal_expected"]
    assert result["is_refusal_predicted"]


def test_parallel_order_is_ignored():
    response = {
        "function_calls": [
            {
                "name": "set_thermostat",
                "arguments": {
                    "temperature": 22,
                },
            },
            {
                "name": "control_lights",
                "arguments": {
                    "room": "living_room",
                    "action": "on",
                },
            },
        ]
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "living_room",
                "action": "on",
            },
        },
        {
            "name": "set_thermostat",
            "arguments": {
                "temperature": 22,
            },
        },
    ]

    result = evaluate_response(response, expected)

    assert result["tool_correct"]
    assert result["arguments_correct"]
    assert result["exact_call_correct"]

    assert result["extra_tools"] == []
    assert result["missing_tools"] == []


def test_extra_tool_is_detected():
    response = {
        "function_calls": [
            {
                "name": "control_lights",
                "arguments": {
                    "room": "study",
                    "action": "dim",
                    "brightness_percent": 30,
                },
            },
            {
                "name": "start_robot_vacuum",
                "arguments": {
                    "action": "start",
                    "room": "living_room",
                },
            },
        ]
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "study",
                "action": "dim",
                "brightness_percent": 30,
            },
        }
    ]

    result = evaluate_response(response, expected)

    assert not result["tool_correct"]
    assert not result["arguments_correct"]
    assert not result["exact_call_correct"]

    assert result["extra_tools"] == [
        "start_robot_vacuum"
    ]

    assert result["missing_tools"] == []

    assert result["num_extra_tools"] == 1
    assert result["num_missing_tools"] == 0


def test_missing_tool_is_detected():
    response = {
        "function_calls": []
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "study",
                "action": "dim",
            },
        }
    ]

    result = evaluate_response(response, expected)

    assert not result["tool_correct"]
    assert not result["arguments_correct"]
    assert not result["exact_call_correct"]

    assert result["extra_tools"] == []
    assert result["missing_tools"] == [
        "control_lights"
    ]


def test_missing_argument_is_detected():
    response = {
        "function_calls": [
            {
                "name": "control_lights",
                "arguments": {
                    "room": "study",
                    "action": "dim",
                },
            }
        ]
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "study",
                "action": "dim",
                "brightness_percent": 30,
            },
        }
    ]

    result = evaluate_response(response, expected)

    # The correct tool was selected.
    assert result["tool_correct"]

    # But the arguments are incomplete.
    assert not result["arguments_correct"]

    # Therefore the complete call is wrong.
    assert not result["exact_call_correct"]


def test_wrong_argument_is_detected():
    response = {
        "function_calls": [
            {
                "name": "set_thermostat",
                "arguments": {
                    "temperature": 25,
                },
            }
        ]
    }

    expected = [
        {
            "name": "set_thermostat",
            "arguments": {
                "temperature": 22,
            },
        }
    ]

    result = evaluate_response(response, expected)

    assert result["tool_correct"]
    assert not result["arguments_correct"]
    assert not result["exact_call_correct"]


def test_parallel_arguments_are_matched_by_tool_name():
    response = {
        "function_calls": [
            {
                "name": "set_thermostat",
                "arguments": {
                    "temperature": 22,
                },
            },
            {
                "name": "control_lights",
                "arguments": {
                    "room": "living_room",
                    "action": "on",
                },
            },
        ]
    }

    expected = [
        {
            "name": "control_lights",
            "arguments": {
                "room": "living_room",
                "action": "on",
            },
        },
        {
            "name": "set_thermostat",
            "arguments": {
                "temperature": 22,
            },
        },
    ]

    result = evaluate_response(response, expected)

    assert result["tool_correct"]
    assert result["arguments_correct"]
    assert result["exact_call_correct"]


def test_duplicate_tool_calls_are_handled():
    response = {
        "function_calls": [
            {
                "name": "set_thermostat",
                "arguments": {
                    "temperature": 21,
                },
            },
            {
                "name": "set_thermostat",
                "arguments": {
                    "temperature": 22,
                },
            },
        ]
    }

    expected = [
        {
            "name": "set_thermostat",
            "arguments": {
                "temperature": 22,
            },
        },
        {
            "name": "set_thermostat",
            "arguments": {
                "temperature": 21,
            },
        },
    ]

    result = evaluate_response(response, expected)

    assert result["tool_correct"]
    assert result["arguments_correct"]
    assert result["exact_call_correct"]


def test_needle_runtime_fields_are_preserved():
    response = {
        "type": "call",
        "success": True,
        "function_calls": [],
        "confidence": None,
        "prefill_tps": 44.9,
        "decode_tps": 34.9,
        "peak_ram_mb": 50.9,
        "validation": {
            "ungrounded": [],
            "negation": False,
        },
    }

    result = evaluate_response(response, [])

    assert result["prefill_tps"] == 44.9
    assert result["decode_tps"] == 34.9
    assert result["peak_ram_mb_engine"] == 50.9

    assert result["validation"] == {
        "ungrounded": [],
        "negation": False,
    }