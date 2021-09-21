def is_available():
    try:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
    except BaseException as err:
        print(f"Error loading module {str(err)}")
        return False
    import habana_frameworks.torch.core as htcore
    return htcore.is_available()
