import os

import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    name="extension_device_import_wrapper",
    sources=[
        str(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "extension_device_import_wrapper.cpp",
            )
        ),
    ],
    extra_cflags=["-g"],
    verbose=True,
)


def empty_strided_extension_device_import(
    size, stride, dtype_opt=None, layout_opt=None, device_opt=None, pin_memory_opt=None
):
    return module._empty_strided_extension_device_import(
        size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt
    )
