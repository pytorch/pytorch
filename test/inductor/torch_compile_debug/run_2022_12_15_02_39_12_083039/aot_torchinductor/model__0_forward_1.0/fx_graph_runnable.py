import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
torch._dynamo.config.load_config(b'\x80\x04\x95\x9d\x07\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x14torch._dynamo.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\rtorch._dynamo\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8c6/home/xiaobing/pytorch-offical/torch/_dynamo/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cN/home/xiaobing/pytorch-offical/torch/_dynamo/__pycache__/config.cpython-38.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x07abspath\x94\x8c\tposixpath\x94h\x1f\x93\x94\x8c\x07dirname\x94h h"\x93\x94\x8c\x0eHAS_REFS_PRIMS\x94\x88\x8c\tlog_level\x94K\x1e\x8c\x0boutput_code\x94\x89\x8c\rlog_file_name\x94N\x8c\x07verbose\x94\x88\x8c\x11output_graph_code\x94\x89\x8c\x12verify_correctness\x94\x89\x8c\x12minimum_call_count\x94K\x01\x8c\x15dead_code_elimination\x94\x88\x8c\x10cache_size_limit\x94K@\x8c\x14specialize_int_float\x94\x88\x8c\x0edynamic_shapes\x94\x89\x8c\x10guard_nn_modules\x94\x89\x8c\x13dynamic_propagation\x94\x88\x8c\x0cnormalize_ir\x94\x89\x8c\x1btraceable_tensor_subclasses\x94\x8f\x94\x8c\x0fsuppress_errors\x94\x89\x8c\x15replay_record_enabled\x94\x88\x8c rewrite_assert_with_torch_assert\x94\x88\x8c\x12print_graph_breaks\x94\x89\x8c\x07disable\x94\x89\x8c*allowed_functions_module_string_ignorelist\x94\x8f\x94(\x8c\x13torch.distributions\x94\x8c\rtorch.testing\x94\x8c\x0ctorch._prims\x94\x8c\rtorch._decomp\x94\x8c\x0btorch._refs\x94\x90\x8c\x16capture_scalar_outputs\x94\x89\x8c\x19enforce_cond_guards_match\x94\x88\x8c\x0coptimize_ddp\x94\x88\x8c\x1araise_on_ctx_manager_usage\x94\x88\x8c\x1craise_on_unsafe_aot_autograd\x94\x88\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\x18error_on_nested_fx_trace\x94\x88\x8c\x08base_dir\x94\x8c\x1e/home/xiaobing/pytorch-offical\x94\x8c\x0edebug_dir_root\x94\x8c@/home/xiaobing/pytorch-offical/test/inductor/torch_compile_debug\x94\x8c)DO_NOT_USE_legacy_non_fake_example_inputs\x94\x89\x8c\x15_AccessLimitingConfig\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x0b__setattr__\x94h\x02\x8c!_AccessLimitingConfig.__setattr__\x94\x93\x94h\x03Nu\x8c\x15_allowed_config_names\x94\x8f\x94(\x8c\x02os\x94h:h2h-h5\x8c\x0c__builtins__\x94hHh9h\x0fhO\x8c\x03sys\x94\x8c\x07logging\x94h.h8h\x01h)h3\x8c\x0brepro_level\x94hCh\x1eh\x1dh*hKh"\x8c\x05torch\x94hMh+hBh(hAhDh\x06h&h,h\x1fh6h0h$h1hE\x8c\x12constant_functions\x94\x8c\nModuleType\x94\x8c\x0eexternal_utils\x94h/\x8c!skipfiles_inline_module_allowlist\x94hJ\x8c\x0brepro_after\x94h\x03h7h\'h\x04hPh%hF\x90\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94hc\x93\x94u.')
torch._inductor.config.load_config(b'\x80\x04\x95\xeb\x08\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x16torch._inductor.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\x0ftorch._inductor\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8c8/home/xiaobing/pytorch-offical/torch/_inductor/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cP/home/xiaobing/pytorch-offical/torch/_inductor/__pycache__/config.cpython-38.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x05debug\x94\x89\x8c\x0bcpp_wrapper\x94\x89\x8c\x03dce\x94\x89\x8c\x0edynamic_shapes\x94\x89\x8c\x14static_weight_shapes\x94\x88\x8c\x0csize_asserts\x94\x88\x8c\x10pick_loop_orders\x94\x88\x8c\x0finplace_buffers\x94\x88\x8c\x11benchmark_harness\x94\x88\x8c\x17realize_reads_threshold\x94K\x04\x8c\x17realize_bytes_threshold\x94M\xd0\x07\x8c\x1brealize_acc_reads_threshold\x94K\x08\x8c\x0ffallback_random\x94\x89\x8c\x12implicit_fallbacks\x94\x88\x8c\rprefuse_nodes\x94\x88\x8c\x0btune_layout\x94\x89\x8c\x11aggressive_fusion\x94\x89\x8c\x0fmax_fusion_size\x94K@\x8c\x1bunroll_reductions_threshold\x94K\x08\x8c\x0ecomment_origin\x94\x89\x8c\tis_fbcode\x94h\x02h3\x93\x94\x8c\x0fcompile_threads\x94K \x8c\x13kernel_name_max_ops\x94K\n\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\rshape_padding\x94\x89\x8c\x11shape_padding_bmm\x94\x88\x8c\x0epermute_fusion\x94\x89\x8c\x1aprofiler_mark_wrapper_call\x94\x89\x8c\x03cpp\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x07threads\x94J\xff\xff\xff\xff\x8c\x0fdynamic_threads\x94\x89\x8c\x07simdlen\x94N\x8c\x0emin_chunk_size\x94M\x00\x10\x8c\x03cxx\x94N\x8c\x03g++\x94\x86\x94\x8c\x15enable_kernel_profile\x94\x89h\x03Nu\x8c\x06triton\x94}\x94(hAh\x02\x8c\ncudagraphs\x94\x89\x8c\x10debug_sync_graph\x94\x89\x8c\x11debug_sync_kernel\x94\x89\x8c\x0bconvolution\x94\x8c\x04aten\x94\x8c\x02mm\x94hP\x8c\x0edense_indexing\x94\x89\x8c\tmax_tiles\x94K\x02\x8c\x08autotune\x94\x89\x8c\x07use_bmm\x94\x89\x8c tiling_prevents_pointwise_fusion\x94\x88\x8c tiling_prevents_reduction_fusion\x94\x88\x8c\x14ordered_kernel_names\x94\x89\x8c\x18descriptive_kernel_names\x94\x88h\x03Nu\x8c\x05trace\x94}\x94(hAh\x02\x8c\x07enabled\x94\x88\x8c\tdebug_log\x94\x88\x8c\x08info_log\x94\x89\x8c\x08fx_graph\x94\x88\x8c\rir_pre_fusion\x94\x88\x8c\x0eir_post_fusion\x94\x88\x8c\x0boutput_code\x94\x88\x8c\rgraph_diagram\x94\x89\x8c\x0fcompile_profile\x94\x89\x8c\nupload_tar\x94Nh\x03Nu\x8c\x15InductorConfigContext\x94}\x94(hAh\x02\x8c\x0f__annotations__\x94}\x94(\x8c\rstatic_memory\x94\x8c\x08builtins\x94\x8c\x04bool\x94\x93\x94\x8c\x0bmatmul_tune\x94hk\x8c\x03str\x94\x93\x94\x8c\x0ematmul_padding\x94hm\x8c\x0ftriton_autotune\x94hm\x8c\ntriton_bmm\x94hm\x8c\ttriton_mm\x94hp\x8c\x12triton_convolution\x94hp\x8c\x17rematerialize_threshold\x94hk\x8c\x03int\x94\x93\x94\x8c\x1brematerialize_acc_threshold\x94hxu\x8c\x05_save\x94h\x02\x8c\x1bInductorConfigContext._save\x94\x93\x94\x8c\x06_apply\x94h\x02\x8c\x1cInductorConfigContext._apply\x94\x93\x94\x8c\x08__init__\x94h\x02\x8c\x1eInductorConfigContext.__init__\x94\x93\x94\x8c\t__enter__\x94h\x02\x8c\x1fInductorConfigContext.__enter__\x94\x93\x94\x8c\x08__exit__\x94h\x02\x8c\x1eInductorConfigContext.__exit__\x94\x93\x94h\x03Nu\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94h\x89\x93\x94u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES

# torch version: 2.0.0a0+gitb7200b4
# torch cuda version: None
# torch git version: b7200b477b231b393a139b23e82b808ddf354601


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        return [convert_element_type]
        
args = [((), (), torch.float32, 'cpu')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro())(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)

