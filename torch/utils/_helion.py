import functools


@functools.cache
def has_helion_package() -> bool:
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def has_helion() -> bool:
    return has_helion_package()
