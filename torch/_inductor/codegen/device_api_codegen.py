
class DeviceApiCodeGen():

    @classmethod
    def py_import_get_raw_stream_as(self, name):
        raise NotImplementedError()

    @classmethod
    def py_set_device(self, device_idx):
        raise NotImplementedError()

    @classmethod
    def py_synchronize(self):
        raise NotImplementedError()

    @classmethod
    def py_DeviceGuard(self, device_idx):
        raise NotImplementedError()

    @classmethod
    def cpp_defAOTStreamGuard(self, name, stream, device_idx):
        raise NotImplementedError()

    @classmethod
    def cpp_defStreamGuard(self, name, stream):
        raise NotImplementedError()

    @classmethod
    def cpp_getStreamFromExternal(self, stream, device_idx):
        raise NotImplementedError()

    @classmethod
    def cpp_defGuard(self, name, device_idx):
        raise NotImplementedError()

