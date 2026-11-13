from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class TestCase:
    id: str
    category: str
    prompt: str
    expected_calls: list[dict[str, Any]]
    description: str = ""
    critical: bool = False
