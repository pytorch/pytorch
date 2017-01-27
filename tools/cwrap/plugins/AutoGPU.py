from . import CWrapPlugin


class AutoGPU(CWrapPlugin):

    def __init__(self, has_self=True, condition=None):
        self.has_self = has_self
        self.condition = condition

    DEFINES = """
#ifdef THC_GENERIC_FILE
#define THCP_AUTO_GPU 1
#else
#define THCP_AUTO_GPU 0
#endif
"""

    def process_option_code_template(self, template, option):
        call = 'THCPAutoGPU __autogpu_guard = THCPAutoGPU(args{});'.format(
            ', (PyObject*)self' if self.has_self else '')

        if self.condition is not None:
            call = "#if {0}\n      {1}\n#endif\n".format(self.condition, call)

        return [call] + template

    def process_full_file(self, code):
        return self.DEFINES + code
