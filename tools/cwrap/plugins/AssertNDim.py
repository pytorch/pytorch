from . import CWrapPlugin
from string import Template


class AssertNDim(CWrapPlugin):

    PRE_CODE_TEMPLATE = Template(
        """if(THTensor_(nDimensionLegacyNoScalars)(LIBRARY_STATE ${arg_op}) != ${dim_value}) {
             THError("Expected argument %s to have %d dimension(s), but has %d",
                     "${op}", ${dim_value}, THTensor_(nDimensionLegacyNoScalars)(LIBRARY_STATE ${arg_op}));
           }
        """)

    def process_option_code_template(self, template, option):
        new_code_pre = []

        for _, arg in enumerate(option['arguments']):
            if 'assert_ndim' not in arg:
                continue

            dim_value = arg.get('assert_ndim')
            op = arg.get('assign_name', arg['name'])
            arg_op = "arg_" + op
            new_code_pre.append(self.PRE_CODE_TEMPLATE.substitute(op=op,
                                                                  arg_op=arg_op,
                                                                  dim_value=dim_value))
            template = new_code_pre + template

        return template
