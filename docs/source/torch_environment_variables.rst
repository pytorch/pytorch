.. _torch_environment_variables:

Torch Environment Variables
===============================
PyTorch uses environment variables to control various settings, such as which
gloo device to use, the number of threads used for parallelism, and the number
of OpenMP threads, and many others. As well, some libraries that PyTorch uses
(e.g., MKL) also use environment variables to control their behavior. This
page lists the environment variables that can be used to configure PyTorch.
Note: There are many environment variables and this list is not exhaustive.


.. toctree::
   :maxdepth: 1

   threading_environment_variables
   cuda_environment_variables
   debugging_environment_variables
   miscellaneous_environment_variables
   logging
