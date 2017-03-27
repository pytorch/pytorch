from . import CWrapPlugin
from string import Template


class ReturnArguments(CWrapPlugin):
    ARGUMENT_RETURN_TEMPLATE = Template("Py_INCREF($arg);\nreturn (PyObject*)($arg);")
    TUPLE_RETURN_TEMPLATE = Template("return PyTuple_Pack($num_args, $args);")

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def get_return_wrapper(self, option):
        if option['return'].startswith('argument '):
            indices = list(map(int, option['return'][len('argument '):].split(',')))
            args = [option['arguments'][idx] for idx in indices]
            accessors = [self.cwrap.get_arg_accessor(arg, option) for arg in args]
            if len(args) == 1:
                return Template(self.ARGUMENT_RETURN_TEMPLATE.safe_substitute(arg=accessors[0]))
            else:
                return Template(self.TUPLE_RETURN_TEMPLATE.safe_substitute(num_args=len(args),
                                                                           args=', '.join(accessors)))
