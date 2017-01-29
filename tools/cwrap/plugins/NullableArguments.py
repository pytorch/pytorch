from . import CWrapPlugin


class NullableArguments(CWrapPlugin):

    def process_single_check(self, code, arg, arg_accessor):
        if 'nullable' in arg and arg['nullable']:
            return '({} || {} == Py_None)'.format(code, arg_accessor)
        return code

    def process_single_unpack(self, code, arg, arg_accessor):
        if 'nullable' in arg and arg['nullable']:
            return '({} == Py_None ? NULL : {})'.format(arg_accessor, code)
        return code
