# Test `@no_type_check`,
# see https://bugs.python.org/issue46571

class NoTypeCheck_Outer:
    class Inner:
        x: int


def NoTypeCheck_function(arg: int) -> int:
    ...
