from __future__ import annotations

import unittest

from tools.strtobool import strtobool


class TestStrtobool(unittest.TestCase):
    def test_truthy(self) -> None:
        for val in (
            "y",
            "Y",
            "yes",
            "YES",
            "Yes",
            "t",
            "T",
            "true",
            "TRUE",
            "on",
            "ON",
            "1",
        ):
            self.assertIs(strtobool(val), True, msg=val)

    def test_falsy(self) -> None:
        for val in (
            "n",
            "N",
            "no",
            "NO",
            "No",
            "f",
            "F",
            "false",
            "FALSE",
            "off",
            "OFF",
            "0",
        ):
            self.assertIs(strtobool(val), False, msg=val)

    def test_invalid_raises(self) -> None:
        for val in ("", "maybe", "2", "yeah", "nope"):
            with self.assertRaises(ValueError, msg=val):
                strtobool(val)


if __name__ == "__main__":
    unittest.main()
