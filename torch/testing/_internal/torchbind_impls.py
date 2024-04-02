import torch
from torch.testing._internal.common_utils import find_library_location, IS_WINDOWS, IS_MACOS, IS_SANDCASTLE, IS_FBCODE

# non-portable load_library call. Skip if IS_MACOS
if not IS_MACOS:
    if IS_WINDOWS:
        lib_file_path = find_library_location("torchbind_test.dll")
        torch.ops.load_library(str(lib_file_path))
    elif IS_SANDCASTLE or IS_FBCODE:
        torch.ops.load_library(
            "//caffe2/test/cpp/jit:test_custom_class_registrations"
        )
    else:
        lib_file_path = find_library_location("libtorchbind_test.so")
        torch.ops.load_library(str(lib_file_path))


    @torch.library.impl_abstract("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        return foo.add_tensor(z)


    @torch.library.impl_abstract("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        return tq.pop()


    @torch.library.impl_abstract("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        return tq.push(x)


    @torch.library.impl_abstract("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq, x):
        return tq.size()
