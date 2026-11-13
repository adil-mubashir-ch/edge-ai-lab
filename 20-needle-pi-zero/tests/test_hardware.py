from needle_bench.hardware import hardware_snapshot

def test_hardware_snapshot():
    x = hardware_snapshot()
    assert "python" in x and "machine" in x
