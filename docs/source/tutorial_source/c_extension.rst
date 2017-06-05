Custom C extensions for pytorch
===============================

Step 1. prepare your C code
---------------------------

First, you have to write your C functions.

Below you can find an example implementation of forward and backward
functions of a module that adds its both inputs.

In your ``.c`` files you can include TH using an ``#include <TH/TH.h>``
directive, and THC using ``#include <THC/THC.h>``.

ffi utils will make sure a compiler can find them during the build.

.. code:: C

    /* src/my_lib.c */
    #include <TH/TH.h>

    int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
    THFloatTensor *output)
    {
        if (!THFloatTensor_isSameSizeAs(input1, input2))
            return 0;
        THFloatTensor_resizeAs(output, input1);
        THFloatTensor_add(output, input1, input2);
        return 1;
    }

    int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
    {
        THFloatTensor_resizeAs(grad_input, grad_output);
        THFloatTensor_fill(grad_input, 1);
        return 1;
    }

There are no constraints on the code, except that you will have to
prepare a single header, which will list all functions want to call from
python.

It will be used by the ffi utils to generate appropriate wrappers.

.. code:: C

    /* src/my_lib.h */
    int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);
    int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

Now, you’ll need a super short file, that will build your custom
extension:

.. code:: python

    # build.py
    from torch.utils.ffi import create_extension
    ffi = create_extension(
    name='_ext.my_lib',
    headers='src/my_lib.h',
    sources=['src/my_lib.c'],
    with_cuda=False
    )
    ffi.build()

Step 2: Include it in your Python code
--------------------------------------

After you run it, pytorch will create an ``_ext`` directory and put
``my_lib`` inside.

Package name can have an arbitrary number of packages preceding the
final module name (including none). If the build succeeded you can
import your extension just like a regular python file.

.. code:: python

    # functions/add.py
    import torch
    from torch.autograd import Function
    from _ext import my_lib


    class MyAddFunction(Function):
        def forward(self, input1, input2):
            output = torch.FloatTensor()
            my_lib.my_lib_add_forward(input1, input2, output)
            return output

        def backward(self, grad_output):
            grad_input = torch.FloatTensor()
            my_lib.my_lib_add_backward(grad_output, grad_input)
            return grad_input

.. code:: python

    # modules/add.py
    from torch.nn import Module
    from functions.add import MyAddFunction

    class MyAddModule(Module):
        def forward(self, input1, input2):
            return MyAddFunction()(input1, input2)


.. code:: python

    # main.py
    import torch.nn as nn
    from torch.autograd import Variable
    from modules.add import MyAddModule

    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__(
                add=MyAddModule(),
            )

        def forward(self, input1, input2):
            return self.add(input1, input2)

    model = MyNetwork()
    input1, input2 = Variable(torch.randn(5, 5)), Variable(torch.randn(5, 5))
    print(model(input1, input2))
    print(input1 + input2)


