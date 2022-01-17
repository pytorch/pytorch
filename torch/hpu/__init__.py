r"""
This package adds support for HPU, use
:func:`is_available()` to determine if your system supports HPU.

"""
def is_available() -> bool:
    try:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
    except BaseException as err:
        print(f"Error loading module {str(err)}")
        return False
    import habana_frameworks.torch.core as htcore
    return htcore.is_available()

def device_count() -> int:
    if is_available():
        import habana_frameworks.torch.core as htcore
        return htcore.get_device_count()
    else:
        return 0

def get_device_type() -> int:
    if is_available():
        import habana_frameworks.torch.core as htcore
        return htcore.get_device_type()
    else:
        return -1
