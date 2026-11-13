from __future__ import annotations
from typing import Any

CORE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "control_lights",
        "description": "Control room lights: turn them on, turn them off, or dim them to a brightness percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "enum": ["kitchen", "living_room", "bedroom", "study"],
                         "description": "Room whose lights should be controlled."},
                "action": {"type": "string", "enum": ["on", "off", "dim"],
                           "description": "Whether to turn lights on, off, or dim them."},
                "brightness_percent": {"type": "integer", "minimum": 0, "maximum": 100,
                                       "description": "Brightness percentage from 0 to 100. Required when dimming."}
            },
            "required": ["room", "action"]
        }
    },
    {
        "name": "set_thermostat",
        "description": "Set the target temperature of the home thermostat in Celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {"type": "integer", "minimum": 10, "maximum": 30,
                                "description": "Target temperature in degrees Celsius."}
            },
            "required": ["temperature"]
        }
    },
    {
        "name": "control_fan",
        "description": "Turn a room fan on or off and optionally select its speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "enum": ["living_room", "bedroom", "study"],
                         "description": "Room containing the fan."},
                "action": {"type": "string", "enum": ["on", "off"],
                           "description": "Turn the fan on or off."},
                "speed": {"type": "string", "enum": ["low", "medium", "high"],
                          "description": "Fan speed."}
            },
            "required": ["room", "action"]
        }
    },
    {
        "name": "control_blinds",
        "description": "Open or close the blinds in a room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "enum": ["kitchen", "living_room", "bedroom", "study"],
                         "description": "Room whose blinds should be controlled."},
                "action": {"type": "string", "enum": ["open", "close"],
                           "description": "Open or close the blinds."}
            },
            "required": ["room", "action"]
        }
    },
    {
        "name": "start_robot_vacuum",
        "description": "Start, stop, or dock the robot vacuum. It can operate in the kitchen, living room, or bedroom.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "dock"],
                           "description": "Vacuum action."},
                "room": {"type": "string", "enum": ["kitchen", "living_room", "bedroom"],
                         "description": "Room for the vacuum operation."}
            },
            "required": ["action", "room"]
        }
    }
]

DISTRACTOR_TOOLS: list[dict[str, Any]] = [
    ("play_music", "Start playing music on a speaker using a free-form search query."),
    ("pause_media", "Pause currently playing media on the speaker."),
    ("resume_media", "Resume currently paused media on the speaker."),
    ("skip_track", "Skip the currently playing music track."),
    ("set_volume", "Set speaker volume from 0 to 100."),
    ("set_timer", "Set a timer using a human-readable duration."),
    ("create_reminder", "Create a reminder with a message and human-readable date and time."),
    ("create_calendar_event", "Create a calendar event with title, start time, and location."),
    ("add_task", "Add an undated task with a title and priority."),
    ("create_note", "Create a note with title and free-form text."),
    ("start_workout", "Start a workout such as running, walking, cycling, swimming, strength, or yoga."),
    ("find_my_phone", "Make the user's phone ring so it can be located."),
    ("log_expense", "Log an expense with amount, category, and merchant."),
    ("log_meal", "Record a meal and classify it as breakfast, lunch, dinner, or snack."),
    ("log_water_intake", "Record water consumption in millilitres."),
    ("log_weight", "Record a person's body weight in kilograms."),
    ("create_contact", "Create a contact with name, phone number, and email address."),
    ("check_battery", "Read the current battery percentage of the device."),
    ("get_weather", "Get current weather information for a city."),
    ("search_contacts", "Search contacts by a person's name."),
    ("search_calendar", "Search calendar events using a query."),
    ("search_music", "Search music catalogues using a query."),
    ("set_alarm", "Create an alarm for a specified human-readable time."),
    ("cancel_timer", "Cancel an existing timer by label."),
    ("set_oven", "Set oven temperature between 50 and 250 Celsius."),
    ("control_coffee_maker", "Start, stop, or warm a coffee maker."),
    ("start_dishwasher", "Start a dishwasher using an eco, heavy, or quick cycle."),
    ("set_cooking_timer", "Set a cooking timer with a label and duration."),
    ("get_appliance_status", "Read the status of an appliance."),
    ("reply_to_notification", "Reply to a notification from a known sender."),
    ("dismiss_notification", "Dismiss a notification from a known sender."),
    ("end_workout", "End the current workout."),
    ("send_message", "Send a text message to a contact."),
    ("read_notifications", "Read the user's current notifications."),
    ("set_screen_brightness", "Set phone screen brightness from 0 to 100."),
    ("enable_wifi", "Turn Wi-Fi on."),
    ("disable_wifi", "Turn Wi-Fi off."),
    ("enable_bluetooth", "Turn Bluetooth on."),
    ("disable_bluetooth", "Turn Bluetooth off."),
    ("set_ringer_volume", "Set phone ringer volume from 0 to 100."),
    ("open_app", "Open an application by name."),
    ("close_app", "Close an application by name."),
    ("take_photo", "Take a photo with the device camera."),
    ("record_video", "Start recording video."),
    ("stop_video", "Stop recording video."),
    ("share_location", "Share the user's current location with a contact."),
    ("get_location", "Return the current device location."),
    ("lock_device", "Lock the device screen."),
    ("unlock_device", "Unlock the device."),
    ("set_do_not_disturb", "Enable or disable do-not-disturb mode."),
    ("get_time", "Return the current local time."),
    ("convert_units", "Convert a quantity between measurement units.")
]

def generic_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": []
        }
    }

ALL_TOOLS = CORE_TOOLS + [generic_schema(n, d) for n, d in DISTRACTOR_TOOLS]

def tools_for_count(count: int) -> list[dict[str, Any]]:
    if count < 5 or count > len(ALL_TOOLS):
        raise ValueError(f"Supported tool counts: 5..{len(ALL_TOOLS)}")
    return ALL_TOOLS[:count]
