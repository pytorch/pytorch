from . import CWrapPlugin
from string import Template


class Broadcast(CWrapPlugin):

    POST_CODE_OUT_PLACE_TEMPLATE = Template(
        """${arg_op_a} = ${arg_op_a}_save;
           ${arg_op_b} = ${arg_op_b}_save;""")

    DEPRECATED_WARNING = \
        """PyErr_WarnEx(PyExc_UserWarning, "${op_a} and ${op_b} not broadcastable, but have the same number of "
                                           "elements.  Falling back to deprecated pointwise behavior.", 1);"""

    # Save and restore passed in arguments in case later plugins use
    PRE_CODE_OUT_PLACE_TEMPLATE = Template(
        """THTensor *${arg_op_a}_save = ${arg_op_a};
           THTensor *${arg_op_b}_save = ${arg_op_b};
           THTensorPtr ${arg_op_a}_guard = THTensor_(new)(LIBRARY_STATE_NOARGS);
           THTensorPtr ${arg_op_b}_guard = THTensor_(new)(LIBRARY_STATE_NOARGS);
           ${arg_op_a} = ${arg_op_a}_guard.get();
           ${arg_op_b} = ${arg_op_b}_guard.get();
           ptrdiff_t ${arg_op_a}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_a}_save);
           ptrdiff_t ${arg_op_b}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_b}_save);
           bool raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_b}_nElem);
           int expanded_err =
               THTensor_(expand2)(LIBRARY_STATE ${arg_op_a}, ${arg_op_b}, ${arg_op_a}_save, ${arg_op_b}_save, raise);
           if (expanded_err != 0 && !raise) {
        """
        + POST_CODE_OUT_PLACE_TEMPLATE.template + "\n"
        + DEPRECATED_WARNING + "\n" +
        """}""")

    POST_CODE_IN_PLACE_TEMPLATE = Template(
        """${arg_op_b} = ${arg_op_b}_save;""")

    PRE_CODE_IN_PLACE_TEMPLATE = Template(
        """THTensor *${arg_op_b}_save = ${arg_op_b};
           THLongStoragePtr ${arg_op_a}_size = THTensor_(newSizeOf)(LIBRARY_STATE ${arg_op_a});
           THTensorPtr ${arg_op_b}_guard = THTensor_(new)(LIBRARY_STATE_NOARGS);
           ${arg_op_b}=${arg_op_b}_guard.get();
           ptrdiff_t ${arg_op_a}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_a});
           ptrdiff_t ${arg_op_b}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_b}_save);
           bool raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_b}_nElem);
           int expanded_err =
               THTensor_(expand)(LIBRARY_STATE ${arg_op_b}, ${arg_op_b}_save, ${arg_op_a}_size.get(), raise);
           if (expanded_err != 0 && !raise) {
        """
        + POST_CODE_IN_PLACE_TEMPLATE.template + "\n"
        + DEPRECATED_WARNING + "\n" +
        """}""")

    def initialize(self, cwrap):
        self.cwrap = cwrap

    # Arguments:
    # [0]: name of tensor to broadcast with
    # [1] inplace (optional).  In place operations only broadcast on second tensor argument
    # [2] fallback (optional).  Will fallback to applying to tensor of equal nElem if broadcast fails
    def process_option_code_template(self, template, option):
        new_code_pre = []
        new_code_post = []
        for _, arg in enumerate(option['arguments']):
            if 'broadcast' not in arg:
                continue

            params = arg.get('broadcast').split(" ")
            op_a =  arg.get('assign_name', arg['name'])
            in_place = "inplace" in params
            raise_errors = "false" if "fallback" in params else "true"

            op_b = params[0]
            arg_op_b = "arg_" + op_b
            arg_op_a = "arg_" + op_a

            if in_place:
                new_code_pre.append(self.PRE_CODE_IN_PLACE_TEMPLATE.substitute(
                    op_a=op_a,
                    op_b=op_b,
                    arg_op_a=arg_op_a,
                    arg_op_b=arg_op_b,
                    raise_errors=raise_errors))
                new_code_pre.append("")

                new_code_post.append(self.POST_CODE_IN_PLACE_TEMPLATE.substitute(
                    arg_op_a=arg_op_a,
                    arg_op_b=arg_op_b,
                    raise_errors=raise_errors))
                new_code_post.append("")
            else:
                new_code_pre.append(self.PRE_CODE_OUT_PLACE_TEMPLATE.substitute(
                    op_a=op_a,
                    op_b=op_b,
                    arg_op_a=arg_op_a,
                    arg_op_b=arg_op_b,
                    raise_errors=raise_errors))
                new_code_pre.append("")

                new_code_post.append(self.POST_CODE_OUT_PLACE_TEMPLATE.substitute(
                    arg_op_a=arg_op_a,
                    arg_op_b=arg_op_b,
                    raise_errors=raise_errors))
                new_code_post.append("")

        template = new_code_pre + template + new_code_post
        return template
