def pytorch_cxx_library(*args, **kwargs):
    cxx_library(*args, **kwargs)


def pytorch_cxx_test(*args, **kwargs):
    cxx_test(*args, **kwargs)


def unified_subdir_glob(*args, **kwargs):
    return subdir_glob(*args, **kwargs)


def third_party(target):
    return '//third_party:{}'.format(target)


def is_fb_internal():
    return False
