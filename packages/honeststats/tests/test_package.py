import honeststats


def test_package_imports_and_has_version() -> None:
    assert isinstance(honeststats.__version__, str)
