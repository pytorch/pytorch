from . import CWrapPlugin

class AutoGPU(CWrapPlugin):

    DEFINES = """
#ifdef THC_GENERIC_FILE
#define THCP_AUTO_GPU 1
#else
#define THCP_AUTO_GPU 0
#endif
"""

    BEFORE_CALL = """
#if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
#endif
"""

    def process_call(self, code, option):
        return self.BEFORE_CALL + code

    def process_full_file(self, code):
        return self.DEFINES + code


