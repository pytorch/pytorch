from . import CWrapPlugin
from string import Template

# Arguments to the Broadcast Plugin:
# broadcast: args_to_broadcast_against [inplace] [fallback]
# [args_to_broadcast_against]: either a single argument (e.g. "arg1") or a comma-seperated
#                              list of two arguments (e.g. "tensor1,tensor2") indicating
#                              arguments to broadcast specified argument (usually "self") against
# [inplace] will generate code for in-place function, which doesn't allow the in-place
#           argument to be broadcast
# [fallback] if tensors aren't broadcastable, preserves "element number" pointwise behavior,
#            where only number of elements need to match, and tensors are viewed as 1-dimensional.
# [dims] specify if the tensors shouldn't be broadcast to a specific tensor or tensors, but a combination
#        of individual dimension sizes of a set of tensors.  For example: addbmm(C,A,B) a.k.a. [C + A @ B]
#        broadcasts C to the first dimension of A and the second dimension of B.  Each dimension is specified as
#        [arg].dim[#] and dimensions are comma-separated.  So, to specify that the tensor should be
#        broadcast to 3-dimensions with sizes:
#        tensor0->size[0] x tensor1->size[1] x tensor2->size[2]
#        you would write:
#        dims:tensor0.dim0,tensor1.dim1,tensor2.dim2
# [types] if the tensors should be of different types than THTensor, specify as X where
#         the actual type to use is THXTensor (i.e. Byte for THByteTensor).  If the type
#         should be THTensor, use 'Real'

# For out of place:
# Two args: expand the two args together
# Three args (fused kernels): (e.g. addcmul) expand all three args together
# Sketch of proof that this is the same:
# consider addcmul, under expansion we want: a + (b * c) = (a + b * c) [all expanded together]
# Let e(i, j) be the expansion of i with j, e(i, j, k) be the expansion of i with j,k
#
# Then a + (b * c) = e(a, e(b,c) * e(c,b)) + e(e(b,c)   * e(c,b), a)
#                  = e(a, e(b,c))          + e(e(b,c)   * e(c,b), a)    (only size matters for second param)
#                  = e(a,b,c)              + e(e(b,c)   * e(c,b), a)    (by associativity of max in expand)
#                  = e(a,b,c)              + e(b,c,a)   * e(c,b,a)      (see L1)
# which is a + b * c all expanded together
#
# L1: Show e(i * j, a) = e(i,a) * e(j,a) where i,j have same size
# Consider any index _{ s_0, ..., s_n}
# e(i * j, a) = (i*j)_{f(s_0), ...,f(s_n)} where f is the expansion of that dimension with a
#             = i_{f(s_0), ..., f(s_n)} * j_{f(s_0), ..., f(s_n)} by definition of pointwise operator
#             = e(i,a) * e(j,a)


class Broadcast(CWrapPlugin):

    # Save and restore passed in arguments in case later plugins use
    POST_TEMPLATE = Template(
        """${arg_op_other} = ${arg_op_other}_save;\n""")

    def getPreArgStringTemplate(self, type=None):
        if type is None:
            ret = """THTensor *${arg_op_other}_save = ${arg_op_other};
                     THTensorPtr ${arg_op_other}_guard(THTensor_(new)(LIBRARY_STATE_NOARGS));\n"""
        else:
            cpu_t = "TH" + type + "Tensor"
            gpu_t = "THCuda" + type + "Tensor"
            ret = ("#if !IS_CUDA\n" +
                   cpu_t + " *${arg_op_other}_save = ${arg_op_other};\n" +
                   cpu_t + "Ptr ${arg_op_other}_guard(" + cpu_t + "_new(LIBRARY_STATE_NOARGS));\n" +
                   "#else\n" +
                   gpu_t + " *${arg_op_other}_save = ${arg_op_other};\n" +
                   "THPPointer<" + gpu_t + "> ${arg_op_other}_guard(\n" + gpu_t + "_new(LIBRARY_STATE_NOARGS));\n" +
                   "#endif\n")
        return Template(ret)

    def getExpandTemplate(self, expand_call, success_code, raise_errors):
        if not raise_errors:
            return Template(
                "bool expand_success = false;\n" +
                "try {\n" +
                expand_call +
                "\nexpand_success = true;\n" +
                "}\n"
                "catch (std::exception &e) {}\n" +
                "if(expand_success) {\n" +
                success_code +
                "\n}\n")
        else:
            return Template(
                expand_call + "\n" +
                success_code + "\n")

    def getOutPlacePreExpand2Template(self, raise_errors):
        expand_code = """expand_outplace2(LIBRARY_STATE ${arg_op_a}_guard.get(), ${arg_op_other}_guard.get(),
                                          ${arg_op_a}, ${arg_op_other},
                                          \"${op_a}\", \"${op_other}\", !${raise_errors});"""
        success_code = """${arg_op_a} = ${arg_op_a}_guard.get();
                          ${arg_op_other} = ${arg_op_other}_guard.get();"""
        return self.getExpandTemplate(expand_code, success_code, raise_errors)

    def getOutPlacePreExpand3Template(self, raise_errors):
        expand_code = """expand_outplace3(LIBRARY_STATE ${arg_op_a}_guard.get(),
                                          ${arg_op_other1}_guard.get(), ${arg_op_other2}_guard.get(),
                                          ${arg_op_a}, ${arg_op_other1}, ${arg_op_other2},
                                          \"${op_a}\", \"${op_other1}\", \"${op_other2}\", !${raise_errors});"""
        success_code = """${arg_op_a} = ${arg_op_a}_guard.get();
                          ${arg_op_other1} = ${arg_op_other1}_guard.get();
                          ${arg_op_other2} = ${arg_op_other2}_guard.get();"""
        return self.getExpandTemplate(expand_code, success_code, raise_errors)

    OUT_PLACE_PRE_EXPAND_PRE_DIM_TEMPLATE = Template(
        """if(THTensor_(nDimension)(LIBRARY_STATE ${arg_op_dim}) <= ${arg_op_dim_value}) {
             THError("Argument %s requires at least %d dimensions, but only has %d",
                     "${op_dim}", ${arg_op_dim_value} + 1, THTensor_(nDimension)(LIBRARY_STATE ${arg_op_dim}));
           }
           long ${arg_op_a}_dim${idx}_size = THTensor_(size)(LIBRARY_STATE ${arg_op_dim}, ${arg_op_dim_value});\n""")

    OUT_PLACE_PRE_EXPAND1_DIM_TEMPLATE = Template(
        """THLongStoragePtr ${arg_op_a}_storage(THLongStorage_newWithSize1(${arg_op_a}_dim0_size));\n""")

    OUT_PLACE_PRE_EXPAND2_DIM_TEMPLATE = Template(
        """THLongStoragePtr ${arg_op_a}_storage(
               THLongStorage_newWithSize2(${arg_op_a}_dim0_size, ${arg_op_a}_dim1_size));\n""")

    OUT_PLACE_PRE_EXPAND3_DIM_TEMPLATE = Template(
        """THLongStoragePtr ${arg_op_a}_storage(
               THLongStorage_newWithSize3(${arg_op_a}_dim0_size, ${arg_op_a}_dim1_size, ${arg_op_a}_dim2_size));\n""")

    def getOutPlacePreExpandPostDimTemplate(self, raise_errors):
        expand_code = """expand(LIBRARY_STATE ${arg_op_a}_guard.get(), ${arg_op_a}, ${arg_op_a}_storage);"""
        success_code = """${arg_op_a} = ${arg_op_a}_guard.get();"""
        return self.getExpandTemplate(expand_code, success_code, raise_errors)

    OUT_PLACE_PRE_TEMPLATE = Template(
        """${code_arg_op_a}${code_arg_op_other1}${code_arg_op_other2}
           ${expand_code}""")

    def getInPlacePreExpand1Template(self, raise_errors):
        expand_code = """expand_inplace1(LIBRARY_STATE ${arg_op_other}_guard.get(), ${arg_op_other}, ${arg_op_a},
                                         \"${op_other}\", \"${op_a}\", !${raise_errors});"""
        success_code = """${arg_op_other} = ${arg_op_other}_guard.get();"""
        return self.getExpandTemplate(expand_code, success_code, raise_errors)

    def getInPlacePreExpand2Template(self, raise_errors):
        expand_code = """expand_inplace2(LIBRARY_STATE ${arg_op_other1}_guard.get(), ${arg_op_other2}_guard.get(),
                                         ${arg_op_other1}, ${arg_op_other2}, ${arg_op_a},
                                         \"${op_other1}\", \"${op_other2}\", \"${op_a}\", !${raise_errors});"""
        success_code = """${arg_op_other1} = ${arg_op_other1}_guard.get();
                          ${arg_op_other2} = ${arg_op_other2}_guard.get();"""
        return self.getExpandTemplate(expand_code, success_code, raise_errors)

    IN_PLACE_PRE_TEMPLATE = Template(
        """${code_arg_op_other1}${code_arg_op_other2}
           ${expand_code}""")

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
            op_a = arg.get('assign_name', arg['name'])
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

            dims_kvs = []
            for p in params:
                if p.startswith("dims:"):
                    assert(raise_errors == "true")
                    if len(dims_kvs) != 0:
                        raise ValueError("multiple specifications of dims")
                    dims = p[len("dims:"):].split(",")
                    for dim in dims:
                        batchdim = dim.split(".")
                        assert len(batchdim) == 2
                        assert batchdim[1].startswith("dim")
                        dim_val = batchdim[1][len("dim"):]
                        dims_kvs.append({"op": batchdim[0], "arg_op": "arg_" + batchdim[0], "val": dim_val})

            assert len(dims_kvs) <= 3
            for p in params[1:]:
                if p != "inplace" and p != "fallback" and not p.startswith("dims:") and not p.startswith("types:"):
                    raise ValueError("invalid parameter {}".format(p))

            type_op_b = None
            type_op_c = None
            for p in params:
                if p.startswith("types:"):
                    if not in_place and len(dims_kvs) > 0:
                        raise ValueError("type specification not supported yet for out-of-place functions "
                                         "that specify explicit dimensions")
                    types = p[len("types:"):].split(",")
                    assert(len(types) == (2 if op_c else 1))
                    type_op_b = None if types[0] == "Real" else types[0]
                    if op_c:
                        type_op_c = None if types[1] == "Real" else types[1]

            op_b_mapping = {
                "op_a": op_a,
                "op_other": op_b,
                "arg_op_a": arg_op_a,
                "arg_op_other": arg_op_b,
                "raise_errors": raise_errors
            }
            op_c_mapping = {
                "op_a": op_a,
                "op_other": op_c,
                "arg_op_a": arg_op_a,
                "arg_op_other": arg_op_c,
                "raise_errors": raise_errors
            }

            if in_place:
                code_arg_op_other1 = self.getPreArgStringTemplate(type=type_op_b).substitute(op_b_mapping)
                code_arg_op_other2 = (
                    self.getPreArgStringTemplate(type=type_op_c).substitute(op_c_mapping) if op_c else "")

                if op_c:
                    expand_code = self.getInPlacePreExpand2Template(raise_errors == "true").substitute(
                        op_b_mapping,
                        op_other1=op_b,
                        op_other2=op_c,
                        arg_op_other1=arg_op_b,
                        arg_op_other2=arg_op_c)
                else:
                    expand_code = self.getInPlacePreExpand1Template(raise_errors == "true").substitute(op_b_mapping)

                new_code_pre.append(self.IN_PLACE_PRE_TEMPLATE.substitute(
                    arg_op_a=arg_op_a,
                    code_arg_op_other1=code_arg_op_other1,
                    code_arg_op_other2=code_arg_op_other2,
                    expand_code=expand_code,
                    raise_errors=raise_errors))
                new_code_pre.append("")

                post_code = self.POST_TEMPLATE.substitute(op_b_mapping)
                if op_c:
                    post_code += self.POST_TEMPLATE.substitute(op_c_mapping)

                new_code_post.append(post_code)
                new_code_post.append("")
            else:
                if len(dims_kvs) != 0:
                    code_arg_op_a = self.getPreArgStringTemplate().substitute(arg_op_other=arg_op_a)
                    code_arg_op_other1 = ""
                    code_arg_op_other2 = ""
                    expand_code = ""
                    for idx, kv in enumerate(dims_kvs):
                        expand_code += self.OUT_PLACE_PRE_EXPAND_PRE_DIM_TEMPLATE.substitute(
                            arg_op_a=arg_op_a,
                            op_dim=kv["op"],
                            arg_op_dim=kv["arg_op"],
                            arg_op_dim_value=kv["val"],
                            idx=idx)

                    if len(dims_kvs) == 1:
                        expand_code += self.OUT_PLACE_PRE_EXPAND1_DIM_TEMPLATE.substitute(
                            arg_op_a=arg_op_a,
                            arg_op_dim0=dims_kvs[0]["arg_op"])
                    elif len(dims_kvs) == 2:
                        expand_code += self.OUT_PLACE_PRE_EXPAND2_DIM_TEMPLATE.substitute(
                            arg_op_a=arg_op_a,
                            arg_op_dim0=dims_kvs[0]["arg_op"],
                            arg_op_dim1=dims_kvs[1]["arg_op"])
                    else:
                        expand_code += self.OUT_PLACE_PRE_EXPAND3_DIM_TEMPLATE.substitute(
                            arg_op_a=arg_op_a,
                            arg_op_dim0=dims_kvs[0]["arg_op"],
                            arg_op_dim1=dims_kvs[1]["arg_op"],
                            arg_op_dim2=dims_kvs[2]["arg_op"])
                    expand_code += self.getOutPlacePreExpandPostDimTemplate(raise_errors == "true").substitute(
                        arg_op_a=arg_op_a,
                        raise_errors=raise_errors)
                    post_code = self.POST_TEMPLATE.substitute(arg_op_other=arg_op_a)

                else:
                    code_arg_op_a = self.getPreArgStringTemplate().substitute(arg_op_other=arg_op_a)
                    code_arg_op_other1 = self.getPreArgStringTemplate(type=type_op_b).substitute(op_b_mapping)
                    code_arg_op_other2 = (self.getPreArgStringTemplate(type=type_op_c).substitute(op_c_mapping)
                                          if op_c else "")

                    if op_c:
                        expand_code = self.getOutPlacePreExpand3Template(raise_errors == "true").substitute(
                            op_b_mapping,
                            op_other1=op_b,
                            op_other2=op_c,
                            arg_op_other1=arg_op_b,
                            arg_op_other2=arg_op_c)

                    else:
                        expand_code = self.getOutPlacePreExpand2Template(
                            raise_errors == "true").substitute(op_b_mapping)

                    post_code = self.POST_TEMPLATE.substitute(arg_op_other=arg_op_a)
                    post_code += self.POST_TEMPLATE.substitute(op_b_mapping)
                    post_code += self.POST_TEMPLATE.substitute(op_c_mapping) if op_c else ""

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
