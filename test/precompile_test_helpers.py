# Owner(s): ["module: dynamo"]
"""Module-level state for test_precompile's residual-global tests: functions
captured in test_precompile reach these through an inlined helper or a module
attribute, which must be rejected at capture rather than mis-served."""

A = 0
B = 0


class _Config:
    def __init__(self):
        self.value = 0


CONFIG = _Config()


def bump_a():
    global A
    A += 1


def bump_a_and_b():
    global A, B
    A = 1
    B = 2
