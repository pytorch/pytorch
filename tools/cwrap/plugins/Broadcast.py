from . import CWrapPlugin
from string import Template


class Broadcast(CWrapPlugin):
    DEPRECATED_WARNING = \
        """PyErr_WarnEx(PyExc_UserWarning, "${op_a} and ${op_other} not broadcastable, but have the same number of "
                                           "elements.  Falling back to deprecated pointwise behavior.", 1);"""

    # Save and restore passed in arguments in case later plugins use
    POST_TEMPLATE = Template(
        """${arg_op_other} = ${arg_op_other}_save;\n""")

    PRE_ARG_OP_OTHER_TEMPLATE = Template(
        """THTensor *${arg_op_other}_save = ${arg_op_other};
           THTensorPtr ${arg_op_other}_guard = THTensor_(new)(LIBRARY_STATE_NOARGS);
           ${arg_op_other}=${arg_op_other}_guard.get();
           ptrdiff_t ${arg_op_other}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_other}_save);
        """)

    OUT_PLACE_PRE_EXPAND2_TEMPLATE = Template(
        """bool ${arg_op_other}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other}_nElem);
           int ${arg_op_other}_err =
             THTensor_(expand2)(LIBRARY_STATE ${arg_op_a}, ${arg_op_other}, ${arg_op_a}_save, ${arg_op_other}_save, ${arg_op_other}_raise);
           if (${arg_op_other}_err != 0 && !${arg_op_other}_raise) {
             ${post_code}"""
             + DEPRECATED_WARNING + "\n" +
        """}""")

    DEPRECATED_WARNING3 = \
        """PyErr_WarnEx(PyExc_UserWarning, "${op_a}, ${op_other1}, and ${op_other2}  not broadcastable, but have the same number of "
                                           "elements.  Falling back to deprecated pointwise behavior.", 1);"""

    OUT_PLACE_PRE_EXPAND3_TEMPLATE = Template(
        """bool ${arg_op_other1}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other1}_nElem);
           bool ${arg_op_other2}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other2}_nElem);
           int ${arg_op_a}_err =
             THTensor_(expand3)(LIBRARY_STATE ${arg_op_a}, ${arg_op_other1}, ${arg_op_other2},
                                ${arg_op_a}_save, ${arg_op_other1}_save, ${arg_op_other2}_save,
                                ${arg_op_other1}_raise || ${arg_op_other2}_raise);
           if (${arg_op_a}_err != 0 && !${arg_op_other1}_raise && !${arg_op_other2}_raise) {
             ${post_code}"""
             + DEPRECATED_WARNING3 + "\n"
        """}""")

    OUT_PLACE_PRE_TEMPLATE = Template(
        """${code_arg_op_a}
           ${code_arg_op_other1}
           ${code_arg_op_other2}
           ${expand_code}
        """)

    IN_PLACE_PRE_EXPAND1_TEMPLATE = Template(
        """bool ${arg_op_other}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other}_nElem);
           int ${arg_op_other}_err =
             !skip_expand && THTensor_(expand)(LIBRARY_STATE ${arg_op_other}, ${arg_op_other}_save, ${arg_op_a}_size.get(), ${arg_op_other}_raise);
           if (${arg_op_other}_err != 0 && !${arg_op_other}_raise) {
             skip_expand = true; // don't do further expansions
             ${post_code}"""
            + DEPRECATED_WARNING + "\n" +
        """}""")

    IN_PLACE_PRE_EXPAND2_TEMPLATE = Template(
        """bool ${arg_op_other1}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other1}_nElem);
           bool ${arg_op_other2}_raise = ${raise_errors} || (${arg_op_a}_nElem != ${arg_op_other2}_nElem);
           int ${arg_op_other1}_err =
             !skip_expand && THTensor_(expand)(LIBRARY_STATE ${arg_op_other1}, ${arg_op_other1}_save, ${arg_op_a}_size.get(), ${arg_op_other1}_raise || ${arg_op_other2}_raise);
           if (${arg_op_other1}_err != 0 && !${arg_op_other1}_raise && ${arg_op_other2}_raise) {
             skip_expand = true; // don't do further expansions
             ${post_code}"""
            + DEPRECATED_WARNING3 + "\n" +
        """}
          int ${arg_op_other2}_err =
            !skip_expand && THTensor_(expand)(LIBRARY_STATE ${arg_op_other2}, ${arg_op_other2}_save, ${arg_op_a}_size.get(), ${arg_op_other1}_raise || ${arg_op_other2}_raise);
           if (${arg_op_other2}_err != 0 && !${arg_op_other1}_raise && ${arg_op_other2}_raise) {
             skip_expand = true; // don't do further expansions
             ${post_code}"""
         + DEPRECATED_WARNING3 + "\n" +
        """}""")

    IN_PLACE_PRE_TEMPLATE = Template(
        """THLongStoragePtr ${arg_op_a}_size = THTensor_(newSizeOf)(LIBRARY_STATE ${arg_op_a});
            ptrdiff_t ${arg_op_a}_nElem = THTensor_(nElement)(LIBRARY_STATE ${arg_op_a});
            bool skip_expand = false;
            ${code_arg_op_other1}
            ${code_arg_op_other2}
            ${expand_code}
        """)

    def initialize(self, cwrap):
        self.cwrap = cwrap

    # Arguments:
    # [0]: name of tensor to broadcast with (possibly two comma separated)
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

            param_others = params[0].split(",")
            if len(param_others) > 2:
                raise ValueError('Broadcast only supports up to 2 secondary parameters')
            op_b = param_others[0]
            op_c = param_others[1] if len(param_others) == 2 else None
            arg_op_b = "arg_" + op_b
            arg_op_a = "arg_" + op_a
            arg_op_c = ("arg_" + op_c) if op_c else None

            op_b_mapping = {
                "op_a":op_a,
                "op_other":op_b,
                "arg_op_a":arg_op_a,
                "arg_op_other":arg_op_b,
                "raise_errors":raise_errors
            }
            op_c_mapping = {
                "op_a":op_a,
                "op_other":op_c,
                "arg_op_a":arg_op_a,
                "arg_op_other":arg_op_c,
                "raise_errors":raise_errors
            }

            if in_place:
                code_arg_op_other1 = self.PRE_ARG_OP_OTHER_TEMPLATE.substitute(op_b_mapping)
                code_arg_op_other2 = self.PRE_ARG_OP_OTHER_TEMPLATE.substitute(op_c_mapping) if op_c else ""


                post_code = self.POST_TEMPLATE.substitute(op_b_mapping)
                if op_c:
                    post_code += self.POST_TEMPLATE.substitute(op_c_mapping)

                if op_c:
                    expand_code = self.IN_PLACE_PRE_EXPAND2_TEMPLATE.substitute(
                        op_b_mapping,
                        op_other1=op_b,
                        op_other2=op_c,
                        arg_op_other1=arg_op_b,
                        arg_op_other2=arg_op_c,
                        post_code=post_code)
                else:
                    expand_code = self.IN_PLACE_PRE_EXPAND1_TEMPLATE.substitute(op_b_mapping, post_code=post_code)

                new_code_pre.append(self.IN_PLACE_PRE_TEMPLATE.substitute(
                    arg_op_a=arg_op_a,
                    code_arg_op_other1=code_arg_op_other1,
                    code_arg_op_other2=code_arg_op_other2,
                    expand_code=expand_code,
                    raise_errors=raise_errors))
                new_code_pre.append("")

                new_code_post.append(post_code)
                new_code_post.append("")
            else:
                code_arg_op_a = self.PRE_ARG_OP_OTHER_TEMPLATE.substitute(arg_op_other=arg_op_a)
                code_arg_op_other1 = self.PRE_ARG_OP_OTHER_TEMPLATE.substitute(op_b_mapping)
                code_arg_op_other2 = self.PRE_ARG_OP_OTHER_TEMPLATE.substitute(op_c_mapping) if op_c else ""

                post_code = self.POST_TEMPLATE.substitute(arg_op_other=arg_op_a)
                post_code += self.POST_TEMPLATE.substitute(op_b_mapping)
                post_code += self.POST_TEMPLATE.substitute(op_c_mapping) if op_c else ""

                if op_c:
                    expand_code = self.OUT_PLACE_PRE_EXPAND3_TEMPLATE.substitute(
                        op_b_mapping,
                        op_other1=op_b,
                        op_other2=op_c,
                        arg_op_other1=arg_op_b,
                        arg_op_other2=arg_op_c,
                        post_code=post_code)
                else:
                    expand_code = self.OUT_PLACE_PRE_EXPAND2_TEMPLATE.substitute(op_b_mapping, post_code=post_code)

                new_code_pre.append(self.OUT_PLACE_PRE_TEMPLATE.substitute(
                    code_arg_op_a=code_arg_op_a,
                    code_arg_op_other1=code_arg_op_other1,
                    code_arg_op_other2=code_arg_op_other2,
                    expand_code=expand_code))
                new_code_pre.append("")

                new_code_post.append(post_code)
                new_code_post.append("")

        template = new_code_pre + template + new_code_post
        return template
