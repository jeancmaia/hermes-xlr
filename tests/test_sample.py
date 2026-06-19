import hermes_nim_xlr


def test_package_imports():
    assert hermes_nim_xlr is not None


def test_fixture_wiring(package_name: str):
    assert package_name == "hermes_nim_xlr"
