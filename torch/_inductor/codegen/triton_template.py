import logging
import os

import sympy

from .. import config, ir
from ..virtualized import V
from .common import IndentedBuffer
from .triton import TritonKernel

log = logging.getLogger((__name__))
template_dict = {ir.Convolution: "triton_conv", ir.MatrixMultiply: "triton_mm"}


class TritonTemplateKernel(TritonKernel):
    def __init__(self, node: ir.ExternKernel, *groups):
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        self.node = node
        self.template_name = template_dict[type(node)]
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(__file__)),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        pid_cache = {}
        if isinstance(node, ir.Convolution):
            pid_cache = {
                "tl.program_id(0)": "pid_nhw",
                "tl.program_id(1)": "pid_k",
            }
            self.map_args()
            KERNEL_H = self.args_dict["KERNEL_H"]
            KERNEL_W = self.args_dict["KERNEL_W"]
            padding_h = self.args_dict["padding_h"]
            padding_w = self.args_dict["padding_w"]
            if ((KERNEL_H == "1" and KERNEL_W == "1")) or (
                (padding_h == "0") and (padding_w == "0")
            ):
                self.template_name += "_delta_x"
            else:
                self.template_name += "_delta_x_hwc"
        elif isinstance(node, ir.MatrixMultiply):
            pid_cache = {
                "tl.program_id(0)": "pid_m",
                "tl.program_id(1)": "pid_n",
            }

        self.template = env.get_template(self.template_name + ".j2")
        super(TritonTemplateKernel, self).__init__(*groups, pid_cache=pid_cache)

    def rename_vars(self):
        for k, v in self.inout_dict.items():
            self.args.output_buffers[v] = k
        if isinstance(self.node, ir.Convolution):
            self.cse.store_cache[self.inout_dict["y"]] = "acc"
        elif isinstance(self.node, ir.MatrixMultiply):
            self.cse.store_cache[self.inout_dict["C"]] = "acc"

    def assign_block_numel(self):
        code = IndentedBuffer()
        if isinstance(self.node, ir.Convolution):
            code.writeline("XBLOCK: tl.constexpr = BLOCK_M")
            code.writeline("YBLOCK: tl.constexpr = BLOCK_N")
            code.writeline(
                "xnumel = BATCH * (OUT_H + 2 * output_padding_h) * (OUT_W + 2 * output_padding_w)"
            )
            code.writeline("ynumel = KERNEL_N")
        elif isinstance(self.node, ir.MatrixMultiply):
            code.writeline("XBLOCK: tl.constexpr = BLOCK_M")
            code.writeline("YBLOCK: tl.constexpr = BLOCK_N")
            code.writeline("xnumel = M")
            code.writeline("ynumel = N")

        return code

    def indexing(self, index: sympy.Expr, copy_shape=None, dense_indexing=True):
        # use dense_indexing for TritonTemplateKernel to avoid map::at error
        return super().indexing(
            index, copy_shape=copy_shape, dense_indexing=dense_indexing
        )

    def codegen_body(
        self, name, fuse, could_remove_kernel_buf, kernel_buf_replace_name
    ):
        """
        put render_variables into the template
        to generate the final code
        """
        # get extra_argdefs from fusable triton kernels
        self.extra_argdefs = []
        self.extra_call_args = []
        argdefs, call_args, _ = self.args.python_argdefs()
        # add extra args if it is different from
        # current TritonTemplateKernel args
        for (argdef, call_arg) in zip(argdefs, call_args):
            if (
                argdef not in self.inout_dict.keys()
                and argdef not in self.args_dict.keys()
            ):
                self.extra_argdefs.append(argdef)
                self.extra_call_args.append(call_arg)

        if could_remove_kernel_buf:
            if isinstance(self.node, ir.Convolution):
                self.inout_dict.pop("y")
            elif isinstance(self.node, ir.MatrixMultiply):
                self.inout_dict.pop("C")
        self.template_inout_argdefs = list(self.inout_dict.keys())

        if kernel_buf_replace_name is not None:
            idx = self.extra_call_args.index(kernel_buf_replace_name)
            kernel_buf_replace_def = self.extra_argdefs[idx]

        super().codegen_body()
        self.pointwise_code = IndentedBuffer()
        self.pointwise_code.splice(self.assign_block_numel())
        self.pointwise_code.splice(self.body)
        render_dict = {}
        render_dict["kernel_name"] = name
        render_dict["template_inout_argdefs"] = self.template_inout_argdefs
        render_dict["extra_argdefs"] = self.extra_argdefs
        render_dict["pointwise_code"] = self.pointwise_code.getvalue() if fuse else None
        render_dict["keep_store"] = not could_remove_kernel_buf
        render_dict["out_def"] = (
            self.out_def() if not could_remove_kernel_buf else kernel_buf_replace_def
        )
        self.body = self.template.render(render_dict) + "\n"

    def out_def(self):
        if isinstance(self.node, ir.Convolution):
            return "y"
        elif isinstance(self.node, ir.MatrixMultiply):
            return "C"

    def codegen_kernel(
        self,
        name=None,
        fuse=False,
        could_remove_kernel_buf=False,
        kernel_buf_replace_name=None,
    ):

        code = IndentedBuffer()

        self.codegen_body(name, fuse, could_remove_kernel_buf, kernel_buf_replace_name)
        code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")

        return wrapper.getvalue()

    def map_args(self):
        """
        map the constant args or
        kernel[grid](..., IN_C, IN_H, IN_W, strides,...)
        """
        (
            self.inout_dict,
            self.args_dict,
            self.const_dict,
            self.other_dict,
        ) = self.node.map_args()

    def precompute(self, wrapper, kernel_name):
        """
        some triton kernels needs host precompute tensor
        for example, triton_conv needs precompte delta_x_ptr
        """
        if isinstance(self.node, ir.Convolution):
            if self.const_dict["CONV1X1_NHWC"] == "False":
                IN_C = self.args_dict["IN_C"]
                KERNEL_H = self.args_dict["KERNEL_H"]
                KERNEL_W = self.args_dict["KERNEL_W"]
                dilation_h = self.args_dict["dilation_h"]
                dilation_w = self.args_dict["dilation_w"]
                stride_wc = self.args_dict["stride_wc"]
                stride_wh = self.args_dict["stride_wh"]
                stride_ww = self.args_dict["stride_ww"]
                stride_xc = self.args_dict["stride_xc"]
                stride_xh = self.args_dict["stride_xh"]
                stride_xw = self.args_dict["stride_xw"]
                device = self.other_dict["device"]
                if self.template_name == "triton_conv_delta_x":
                    assert "delta_x_ptr" not in self.args_dict.keys()
                    self.args_dict["delta_x_ptr"] = f"delta_x_{kernel_name}"
                    wrapper.writeline(
                        f"from {config.inductor_import}.triton_ops import _conv"
                    )
                    wrapper.writeline(
                        f"delta_x_{kernel_name} = _conv._delta_x_ptr("
                        f"{IN_C}, {KERNEL_H}, {KERNEL_W}, "
                        f"{dilation_h}, {dilation_w}, "
                        f"{stride_wc}, {stride_wh}, {stride_ww}, "
                        f"{stride_xc}, {stride_xh}, {stride_xw}, {device})"
                    )
                # triton_conv_delta_x_hwc
                else:
                    assert "delta_xh_ptr" not in self.args_dict.keys()
                    assert "delta_xw_ptr" not in self.args_dict.keys()
                    assert "delta_xc_ptr" not in self.args_dict.keys()
                    self.args_dict["delta_xh_ptr"] = f"delta_xh_{kernel_name}"
                    self.args_dict["delta_xw_ptr"] = f"delta_xw_{kernel_name}"
                    self.args_dict["delta_xc_ptr"] = f"delta_xc_{kernel_name}"
                    wrapper.writeline(
                        f"from {config.inductor_import}.triton_ops import _conv"
                    )
                    wrapper.writeline(
                        f"delta_xh_{kernel_name}, delta_xw_{kernel_name}, delta_xc_{kernel_name}"
                        f" = _conv._delta_x_ptr_hwc("
                        f"{IN_C}, {KERNEL_H}, {KERNEL_W}, "
                        f"{dilation_h}, {dilation_w}, "
                        f"{stride_wc}, {stride_wh}, {stride_ww}, "
                        f"{stride_xc}, {stride_xh}, {stride_xw}, {device})"
                    )

            # else, delta_x_ptr is None
            else:
                assert "delta_x_ptr" not in self.args_dict.keys()
                self.args_dict["delta_x_ptr"] = "None"
        return

    def gen_grid(self, name):
        code = IndentedBuffer()
        if isinstance(self.node, ir.Convolution):
            BATCH = self.args_dict["BATCH"]
            OUT_H = self.args_dict["OUT_H"]
            OUT_W = self.args_dict["OUT_W"]
            KERNEL_N = self.args_dict["KERNEL_N"]
            with code.indent():
                code.splice(
                    f"""
                    def grid_{name}(META):
                        return (
                            triton.cdiv({BATCH} * {OUT_H} * {OUT_W}, META["BLOCK_M"]),
                            triton.cdiv({KERNEL_N}, META["BLOCK_N"]),
                        )
                    """
                )
        if isinstance(self.node, ir.MatrixMultiply):
            M = self.args_dict["M"]
            N = self.args_dict["N"]
            with code.indent():
                code.splice(
                    f"""
                    def grid_{name}(META):
                        return (
                            triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]),
                            META["SPLIT_K"],
                        )
                    """
                )
        return code.getvalue()

    def call_kernel(self, wrapper, name: str):
        # gen code to call kernel
        # e.g.
        # def grid(META):
        #     return (...)
        # kernel1[grid](arg0, arg1, ...)
        extra_args = ", ".join(self.extra_call_args)
        self_args = ", ".join({**self.inout_dict, **self.args_dict}.values())
        self_const_kwargs = ", ".join(f"{k}={v}" for k, v in self.const_dict.items())
        args = self_args + (
            ", " + extra_args if extra_args and len(extra_args) > 0 else ""
        )
        args_kwargs = args + ", " + self_const_kwargs
        wrapper.writeline(self.gen_grid(name))
        wrapper.writeline(f"{name}[grid_{name}]({args_kwargs})")


def should_use_template(node: ir.ExternKernel):
    template_kernels = [ir.Convolution, ir.MatrixMultiply]
    if type(node) in template_kernels and ir.is_triton(node.get_device()):
        if isinstance(node, ir.Convolution):
            return node.kernel != "aten.convolution"
        elif isinstance(node, ir.MatrixMultiply):
            return node.kernel != "aten.mm.out"
    return False


def template_can_fuse(snode1, snode2):
    assert snode1.is_template()
    if snode1.group != snode2.group:
        return False
    tiling = snode1.get_nodes()[0].node.get_template_tiling()
    for node in snode2.get_nodes():
        if not TritonKernel.is_compatible(tiling, node.get_ranges()):
            return False
    return True


def template_codegen(scheduler, scheduler_node, epilogue):
    """
    codegen function for triton templates
    scheduler: Scheduler
    scheduler_node: ExternKernelSchedulerNode
    """
    log.debug("template_codegen: %s -- %s", scheduler_node, epilogue)

    wrapper = V.graph.wrapper_code
    _, groups = scheduler_node.group

    with TritonTemplateKernel(
        scheduler_node.node, *scheduler_node.node.get_template_tiling()
    ) as kernel:
        # map const args/ shape/ strides to kernel args
        kernel.map_args()
        # set self.args name to match the TritonTemplateKernel's args names
        kernel.rename_vars()
        # scheduler.pop_group will keep iterating all reachable fusable SchedulerNodes
        assert type(kernel.node) in template_dict.keys()

        kernel.store_buffer_names.add(scheduler_node.get_name())

        for node in epilogue:
            node.mark_run()
            node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

    could_remove_kernel_buf = (
        kernel.args.output_buffers[scheduler_node.get_name()] == "REMOVED"
    )
    kernel_buf_replace_name = None
    if could_remove_kernel_buf:
        for node in epilogue:
            if kernel.args.output_buffers[node.get_name()] != "REMOVED":
                kernel_buf_replace_name = node.get_name()
                break
        assert kernel_buf_replace_name is not None

    kernel_name = "triton_template_" + wrapper.next_kernel_suffix()
    # code gen kernel
    wrapper.header.splice(
        kernel.codegen_kernel(
            kernel_name,
            bool(epilogue),
            could_remove_kernel_buf,
            kernel_buf_replace_name,
        )
    )
    # gen precompute tensor (like delta_x_ptr) if needed
    kernel.precompute(wrapper, kernel_name)
    # code gen call to kernel
    kernel.call_kernel(wrapper, kernel_name)
