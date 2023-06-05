from unittest import mock


def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigMixin):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)
