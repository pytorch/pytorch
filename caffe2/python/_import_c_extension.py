import atexit
import logging
import sys

# We will first try to load the gpu-enabled caffe2. If it fails, we will then
# attempt to load the cpu version. The cpu backend is the minimum required, so
# if that still fails, we will exit loud.
try:
    from .libcaffe2_python_gpu import *
    has_gpu_support = True
except ImportError as e:
    logging.error(
        'This caffe2 python run does not have GPU support. Error: {0}'
        .format(str(e)))
    has_gpu_support = False
    try:
        from .libcaffe2_python_cpu import *
    except ImportError as e:
        logging.critical(
            'Cannot load caffe2.python. Error: {0}'.format(str(e)))
        sys.exit(1)

# libcaffe2_python contains a global Workspace that we need to properly delete
# when exiting. Otherwise, cudart will cause segfaults sometimes.
atexit.register(OnModuleExit)
