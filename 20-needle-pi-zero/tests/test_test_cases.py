from needle_bench.benchmark import load_test_cases
from needle_bench.tools import ALL_TOOLS, tools_for_count

def test_unique_case_ids():
    cases = load_test_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))

def test_expected_tools_exist():
    names = {x["name"] for x in ALL_TOOLS}
    for c in load_test_cases():
        for call in c.expected_calls:
            assert call["name"] in names

def test_tool_counts():
    assert [len(tools_for_count(x)) for x in (5,10,25,50)] == [5,10,25,50]
