# mypy: allow-untyped-defs
import collections


counters: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
