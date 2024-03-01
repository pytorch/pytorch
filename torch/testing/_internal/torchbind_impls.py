import torch
from torch.testing._internal.common_utils import find_library_location, IS_WINDOWS

if IS_WINDOWS:
    lib_file_path = find_library_location("torchbind_test.dll")
    torch.ops.load_library(str(lib_file_path))
else:
    lib_file_path = find_library_location("libtorchbind_test.so")
    torch.ops.load_library(str(lib_file_path))


@torch.library.impl_abstract("_TorchScriptTesting::takes_foo")
def fake_takes_foo(foo, z):
    return foo.add_tensor(z)
