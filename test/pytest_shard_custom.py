"""
Custom pytest shard plugin
https://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py#L1
Modifications:
* shards are now 1 indexed instead of 0 indexed
* option for printing items in shard
"""

import hashlib

from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser):
    """Add options to control sharding."""
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-id", dest="shard_id", type=int, default=1, help="Number of this shard."
    )
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    group.addoption(
        "--print-items",
        dest="print_items",
        action="store_true",
        default=False,
        help="Print out the items being tested in this shard.",
    )


class PytestShardPlugin:
    def __init__(self, config):
        self.config = config

    def pytest_report_collectionfinish(self, config, items) -> str:
        """Log how many and which items are tested in this shard."""
        msg = f"Running {len(items)} items in this shard"
        if config.getoption("print_items"):
            msg += ": " + ", ".join([item.nodeid for item in items])
        return msg

    def sha256hash(self, x: str) -> int:
        return int.from_bytes(hashlib.sha256(x.encode()).digest(), "little")

    def filter_items_by_shard(self, items, shard_id: int, num_shards: int):
        """Computes `items` that should be tested in `shard_id` out of `num_shards` total shards."""
        new_items = [
            item
            for item in items
            if self.sha256hash(item.nodeid) % num_shards == shard_id - 1
        ]
        return new_items

    def pytest_collection_modifyitems(self, config, items):
        """Mutate the collection to consist of just items to be tested in this shard."""
        shard_id = config.getoption("shard_id")
        shard_total = config.getoption("num_shards")
        if shard_id < 1 or shard_id > shard_total:
            raise ValueError(
                f"{shard_id} is not a valid shard ID out of {shard_total} total shards"
            )

        items[:] = self.filter_items_by_shard(items, shard_id, shard_total)
