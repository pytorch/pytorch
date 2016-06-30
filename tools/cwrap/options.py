import math
from copy import deepcopy
from itertools import chain

from .utils import argfilter
from .config import *

def make_option(name, rettype, flags):
    if rettype == 'CUSTOM':
        return CustomTHOption(name, rettype, flags)
    if 'PLAIN_CALL' in flags:
        return PlainOption(name, rettype, flags)
    if 'STORAGE_CALL' in flags:
        return THStorageOption(name, rettype, flags)
    if 'LONG_ARGS' in flags:
        return LongArgsTHOption(name, rettype, flags)
    return THOption(name, rettype, flags)

def argcount(option):
    is_already_provided = argfilter()
    return sum(1 for arg in option.arguments if not is_already_provided(arg))

class Option(object):
    OPTION_CODE = Template("""
      if (PyArg_ParseTuple(args, $format$parse_args)) {
        $expr;
      }\n""")

    def __init__(self, funcname, return_type, flags):
        self.funcname = funcname
        self.flags = flags
        self.return_type = return_type
        self.arguments = []

    def add_argument(self, arg):
        self.arguments.append(arg)

    def insert_argument(self, idx, arg):
        self.arguments.insert(idx, arg)

    def map_arguments(self, fn):
        self.arguments = list(map(fn, self.arguments))

    def is_self_optional(self):
        return 'OPTIONAL_SELF' in self.flags

    def _get_all_args(self):
        """Returns a list containing all arguments that should be passed to a
           wrapped function.

           This is necessary only, because of additional args (some functions
           require allocating new output objects).
        """
        additional_args = ADDITIONAL_ARGS.get(self.return_type, ())
        if isinstance(additional_args, dict):
            arg_iter = deepcopy(self.arguments)
            for k,v in additional_args.items():
                arg_iter.insert(k, v)
        else:
            arg_iter = chain(additional_args, self.arguments)
        return list(arg_iter)

    def _build_argstring(self):
        """Builds a string containing C code with all arguments, comma separated.
        """
        all_args = self._get_all_args()
        def make_arg(arg):
            if arg.type == 'EXPRESSION':
                return arg.name.format(*tuple(a.name for a in all_args))
            return arg.name + ('->cdata' if arg.type in CDATA_TYPES else '')
        return ', '.join(make_arg(arg) for arg in all_args)

    def _make_call(self, argstr):
        raise NotImplementedError

    def generate(self):
        """Generates code implementing one call option
        """
        format_str = self._make_format_str()
        argparse_args = self._argparse_arguments()
        expression = self._make_call(self._build_argstring())
        # This is not only an optimization, but also prevents PyArg_ParseTuple from
        # segfaulting - it doesn't handle args == NULL case.
        if self.num_required_args() == 0:
            return expression + ';'
        return self.OPTION_CODE.substitute({
            'format': format_str,
            'parse_args': argparse_args,
            'expr': expression,
        })

    def _make_format_str(self):
        """Returns a format string for PyArg_ParseTuple.
        """
        is_already_provided = argfilter()
        s = ''.join(FORMAT_STR_MAP[arg.type] for arg in self.arguments \
                        if not is_already_provided(arg))
        return '"' + s + '"'

    def _argparse_arguments(self):
        """Builds a list of variables (and type pointers for type checking) to
        be used with PyArg_ParseTuple.
        """
        is_already_provided = argfilter()
        s = ', '
        for arg in self.arguments:
            if is_already_provided(arg):
                continue
            parsed_type = ARGPARSE_TYPE_CHECK.get(arg.type)
            if parsed_type:
                s += '&' + parsed_type + ', '
            s += '&' + arg.name + ', '
        return s.rstrip()[:-1] # Remove whitespace and trailing comma

    def copy(self):
        return deepcopy(self)

    def signature_hash(self):
        is_already_provided = argfilter()
        s = '#'.join(arg.type for arg in self.arguments if not is_already_provided(arg))
        return s.__hash__()

    def num_required_args(self):
        """Returns a number of unspecified args.

           Iff, the option is variadic, returns infinity.
        """
        return argcount(self)

    def _library_state_macro(self, argstr):
        return 'LIBRARY_STATE' if argstr else 'LIBRARY_STATE_NOARGS'


class PlainOption(Option):
    def _make_call(self, argstr):
        library_state = self._library_state_macro(argstr)
        call = '{}({})'.format(self.funcname, argstr)
        return RETURN_WRAPPER[self.return_type].substitute({'expr': call})


class THOption(Option):
    def _make_call(self, argstr):
        library_state = self._library_state_macro(argstr)
        th_call = 'THTensor_({})({} {})'.format(self.funcname, library_state, argstr)
        return RETURN_WRAPPER[self.return_type].substitute({'expr': th_call})


class THStorageOption(Option):
    def _make_call(self, argstr):
        library_state = self._library_state_macro(argstr)
        th_call = 'THStorage_({})({} {})'.format(self.funcname, library_state, argstr)
        return RETURN_WRAPPER[self.return_type].substitute({'expr': th_call})


class CustomTHOption(Option):
    def _make_call(self, argstr):
        library_state = self._library_state_macro(argstr)
        th_call = 'THTensor_({})({} {})'.format(self.funcname, library_state, argstr)
        return self.flags.format(expr=th_call)


class LongArgsTHOption(THOption):
    OPTION_CODE = Template("""
      if ($checks) {
        THLongStoragePtr _long_args = THPUtils_getLongStorage(args, $ignored_args);
        $parse
        $expr;
      }\n""")

    def generate(self):
        """Generates code implementing one call option
        """
        checks = self._make_checks()
        variable_init = self._make_variable_init()
        expression = self._make_call(self._build_argstring())
        return self.OPTION_CODE.substitute({
            'checks': checks,
            'parse': variable_init,
            'expr': expression,
            'ignored_args': argcount(self),
        })

    def _make_checks(self):
        arg_idx = 0
        check_str = ''
        is_provided = argfilter()
        for arg in self.arguments:
            if is_provided(arg):
                continue
            check_str += ' && ' + TYPE_CHECK[arg.type]('PyTuple_GET_ITEM(args, {})'.format(arg_idx))
            arg_idx += 1
        check_str = '_argcount > ' + str(arg_idx) + check_str
        return check_str

    def _make_variable_init(self):
        init = ''
        arg_idx = 0
        is_provided = argfilter()
        for arg in self.arguments:
            if is_provided(arg):
                continue
            if arg_idx > 0:
                init += '\n    '
            init += arg.name + ' = ({})PyTuple_GET_ITEM(args, {});'.format(arg.type, arg_idx)
            arg_idx += 1
        return init

    def num_required_args(self):
        return math.inf

