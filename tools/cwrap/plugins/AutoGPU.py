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

    def process_option_code_template(self, template, option):
        return [self.BEFORE_CALL] + template

    def process_full_file(self, code):
        return self.DEFINES + code


