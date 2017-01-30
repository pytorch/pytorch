from . import CWrapPlugin
from string import Template


class KwargsPlugin(CWrapPlugin):

    ACCESSOR_TEMPLATE = Template('(__tuplecount > $idx ? PyTuple_GET_ITEM(args, $idx) : __kw_$name)')
    KWARG_ONLY_ACCESSOR_TEMPLATE = Template('__kw_$name')
    CHECK_TEMPLATE = Template('(__tuplecount > $idx || __kw_$name) && $code')
    KWARG_ONLY_CHECK_TEMPLATE = Template('__kw_$name && $code')
    WRAPPER_TEMPLATE = Template("""
    $declarations
    if (kwargs) {
      $lookups
    }
    """)

    def process_declarations(self, declarations):
        # We don't have access to declaration or options in get_arg_accessor
        # and process_single_check, so we have to push the flag down to
        # the args.
        for declaration in declarations:
            if declaration.get('no_kwargs'):
                for option in declaration['options']:
                    for arg in option['arguments']:
                        arg['no_kwargs'] = True
        return declarations

    def get_arg_accessor(self, arg, option):
        if arg.get('no_kwargs'):
            return
        if arg.get('kwarg_only'):
            return self.KWARG_ONLY_ACCESSOR_TEMPLATE.substitute(name=arg['name'])
        return self.ACCESSOR_TEMPLATE.substitute(idx=arg['idx'], name=arg['name'])

    def process_single_check(self, code, arg, arg_accessor):
        if arg.get('no_kwargs'):
            return code
        if arg.get('kwarg_only'):
            return self.KWARG_ONLY_CHECK_TEMPLATE.substitute(name=arg['name'], code=code)
        return self.CHECK_TEMPLATE.substitute(idx=arg['idx'], name=arg['name'], code=code)

    def process_wrapper(self, code, declaration):
        if declaration.get('no_kwargs'):
            return code
        seen_args = set()
        args = []
        for option in declaration['options']:
            for arg in option['arguments']:
                name = arg['name']
                if (not arg.get('ignore_check') and
                        not arg.get('no_kwargs') and
                        name not in seen_args):
                    seen_args.add(name)
                    args.append(name)
        declarations = '\n    '.join(['PyObject *__kw_{} = NULL;'.format(name) for name in args])
        lookups = '\n      '.join(
            ['__kw_{name} = PyDict_GetItemString(kwargs, "{name}");'.format(name=name) for name in args])
        start_idx = code.find('{') + 1
        new_code = self.WRAPPER_TEMPLATE.substitute(declarations=declarations, lookups=lookups)
        return code[:start_idx] + new_code + code[start_idx:]
