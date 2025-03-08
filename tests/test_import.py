def test_widom_insertion_import():
    try:
        from dac_sim.widom_insertion import WidomInsertion
    except Exception as e:
        assert False, f"Import failed: {e}"
