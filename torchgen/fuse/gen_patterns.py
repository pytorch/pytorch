#!/usr/bin/env python3
import os

from torch._inductor import pattern_matcher
from torch._inductor.fx_passes import joint_graph

if __name__ == "__main__":
    # Start by deleting all the existing patterns.
    for file in os.listdir(pattern_matcher.SERIALIZED_PATTERN_PATH):
        if file in ("__init__.py", "__pycache__"):
            continue
        file = pattern_matcher.SERIALIZED_PATTERN_PATH / file
        if file.is_file():
            file.unlink()

    # Now have joint_graph load all known patterns and tell the pattern matcher
    # to serialize the patterns as it goes.
    os.environ["PYTORCH_GEN_PATTERNS"] = "1"
    joint_graph.lazy_init()
