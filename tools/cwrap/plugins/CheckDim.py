from . import CWrapPlugin
from string import Template


class CheckDim(CWrapPlugin):

    PRE_CODE_TEMPLATE = Template(
        """if(THTensor_(nDimension)(LIBRARY_STATE ${arg_op}) != ${dim_value}) {
             THError("Expected argument %s to have %d dimensions, but has %d",
                     "${op}", ${dim_value}, THTensor_(nDimension)(LIBRARY_STATE ${arg_op}));
           }
        """)

    def process_option_code_template(self, template, option):
        new_code_pre = []

        for _, arg in enumerate(option['arguments']):
            if 'checkdim' not in arg:
                continue

            dim_value = arg.get('checkdim')
            op = arg.get('assign_name', arg['name'])
            arg_op = "arg_" + op
            new_code_pre.append(self.PRE_CODE_TEMPLATE.substitute(op=op,
                                                                  arg_op=arg_op,
                                                                  dim_value=dim_value))
            template = new_code_pre + template

        return template
