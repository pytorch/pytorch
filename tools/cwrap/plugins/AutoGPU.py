from . import CWrapPlugin


class AutoGPU(CWrapPlugin):

    def __init__(self, has_self=True, condition=None):
        self.has_self = has_self
        self.condition = condition

    def process_pre_arg_assign(self, template, option):
        if not option.get('auto_gpu', True):
            return template
        call = 'AutoGPU auto_gpu(get_device(args));'
        return [call] + template
