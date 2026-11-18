"""pytest's plugin for the containers of ramjet test. With ``-p tools.ramjet.serial
--ramjet-serial=FILE`` it writes to FILE the node ids of the tests marked
@serialTest, one a line: those run_test.py runs with nothing beside them."""

# Not in the linter image
# pyrefly: ignore [missing-import]
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ramjet-serial",
        metavar="FILE",
        help="write the node ids of the tests marked @serialTest to FILE, one a line",
    )


# First, before -k and the like leave tests out: run_test.py counts those too
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if path := config.getoption("ramjet_serial"):
        marked = [item.nodeid for item in items if item.get_closest_marker("serial")]
        with open(path, "w") as file:
            file.write("\n".join(marked))
