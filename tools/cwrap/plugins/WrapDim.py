from . import CWrapPlugin
from string import Template


class WrapDim(CWrapPlugin):

    NDIM_TEMPLATE = Template(
        """${arg_tensor}->nDimension""")

    CODE_TEMPLATE = Template(
        """THPUtils_assert(${ndim} > 0,
         "dimension specified as %d, but tensor has no dimensions", ${arg_dim});
         THPUtils_assert(${arg_dim} >= -(${ndim}) && ${arg_dim} < (${ndim}),
         "dimension out of range (expected to be in range of [%d, %d], but got %d)",
         -(${ndim}), (${ndim})-1, ${arg_dim});
         if (${arg_dim} < 0) ${arg_dim} += (${ndim});""")

    def initialize(self, cwrap):
        self.cwrap = cwrap

    def process_option_code_template(self, template, option):
        new_code = []
        for i, arg in enumerate(option['arguments']):
            if 'wrap_dim' not in arg:
                continue

            params = arg.get('wrap_dim').split("+")
            arg_tensor = params[0]

            arg_tensor = "arg_" + arg_tensor
            arg_dim = "arg_" + arg.get('assign_name', arg['name'])

            params[0] = self.NDIM_TEMPLATE.substitute(arg_tensor=arg_tensor)
            ndim = "+".join(params)

            new_code.append(self.CODE_TEMPLATE.substitute(
                arg_dim=arg_dim,
                ndim=ndim))
            new_code.append("")

        template = new_code + template
        return template
