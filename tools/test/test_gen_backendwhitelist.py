import os
import tempfile
import unittest
import pathlib
from typing import Optional

import expecttest


from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE  # noqa: F401
from torchgen.gen import (
    gen_source_files, 
    gen_headers, 
    parse_native_yaml, 
    get_grouped_native_functions,
    get_grouped_by_view_native_functions,
    get_custom_build_selector,
    _GLOBAL_PARSE_TAGS_YAML_CACHE,
)
from torchgen.model import (
    DispatchKey,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    is_generic_dispatch_key,
)
from torchgen.utils import (
    FileManager
)

path = os.path.dirname(os.path.realpath(__file__))
torch_xpu_ops_path = os.path.join(path, "../../third_party/torch-xpu-ops/")
xpu_yaml_path= os.path.join(torch_xpu_ops_path, "yaml/native/native_functions.yaml")
template_path = os.path.join(torch_xpu_ops_path, "yaml/templates/")
aoti_dir = os.path.join(path, "xpu_generated")
install_dir = os.path.join(path, "xpu_generated")
class TestGenXPUBackendWhiteList(expecttest.TestCase):
    def setUp(self)-> None:
        yaml_path = xpu_yaml_path
        tags_yaml_path = os.path.join(torch_xpu_ops_path, "yaml/native/tags.yaml")

        selector = get_custom_build_selector(
            None,
            None
        )

        ignore_keys = set()
        ignore_keys.add(DispatchKey.MPS)
        parsed_yaml = parse_native_yaml(yaml_path, tags_yaml_path, ignore_keys)

        whitelist_keys = set({DispatchKey.XPU})
        valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )

        grouped_native_functions = get_grouped_native_functions(native_functions)

        structured_native_functions = [
            g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)
        ]

        native_functions_with_view_groups = get_grouped_by_view_native_functions(
            native_functions
        )
        view_groups = [
            g
            for g in native_functions_with_view_groups
            if isinstance(g, NativeFunctionsViewGroup)
        ]

        core_install_dir = f"{install_dir}/core"
        pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)
        ops_install_dir = f"{install_dir}/ops"
        pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
        aoti_install_dir = f"{aoti_dir}"
        pathlib.Path(aoti_install_dir).mkdir(parents=True, exist_ok=True)

        core_fm = FileManager(install_dir=core_install_dir, template_dir=template_path, dry_run=False)
        cpu_fm = FileManager(install_dir=install_dir, template_dir=template_path, dry_run=False)
        cpu_vec_fm = FileManager(install_dir=install_dir, template_dir=template_path, dry_run=False)
        cuda_fm = FileManager(install_dir=install_dir, template_dir=template_path, dry_run=False)
        ops_fm = FileManager(install_dir=ops_install_dir, template_dir=template_path, dry_run=False)
        aoti_fm = FileManager(install_dir=aoti_install_dir, template_dir=template_path, dry_run=False)

        # Only a limited set of dispatch keys get CPUFunctions.h headers generated
        # for them; this is the set
        functions_keys = {
            DispatchKey.CPU,
            DispatchKey.CUDA,
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
            DispatchKey.Meta,
            DispatchKey.XPU,
        }

        from torchgen.model import dispatch_keys

        dispatch_keys = [
            k
            for k in dispatch_keys
            if (is_generic_dispatch_key(k) or str(k) in ["XPU"])
        ]

        static_dispatch_idx: List[BackendIndex] = []
        static_dispatch_idx = []
        # for key in options.static_dispatch_backend:
        #     dp_key = DispatchKey.parse(key)
        #     if dp_key not in functions_keys:
        #         functions_keys.add(dp_key)

        gen_source_files(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            view_groups=view_groups,
            selector=selector,
            static_dispatch_idx=static_dispatch_idx,
            backend_indices=backend_indices,
            aoti_fm=aoti_fm,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            cpu_vec_fm=cpu_vec_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            whitelist_keys=whitelist_keys,
            rocm=False,
            force_schema_registration=False,
            per_operator_headers=True,
            skip_dispatcher_op_registration=False,
            update_aoti_c_shim=False,
        )

        gen_headers(
            native_functions=native_functions,
            valid_tags=valid_tags,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            whitelist_keys=whitelist_keys,
            rocm=False,
            per_operator_headers=True,      
        )

    def file_has_words(self, keyword, file):
        with open(file, "r") as f:
            for line in f:
                if keyword in line:
                    return True
        return False        

    def test_generated_fils(self):
        assert os.path.exists(os.path.join(install_dir, "ops/as_strided_native.h"))
        # check structure operators
        mul_file = os.path.join(install_dir, "ops/mul_native.h")
        assert os.path.exists(mul_file)
        assert self.file_has_words("struct TORCH_API structured_mul_out ", mul_file)
        # check  unstructured operators
        dropout_file = os.path.join(install_dir, "ops/native_dropout_native.h")
        assert os.path.exists(dropout_file)
        assert self.file_has_words("TORCH_API ::std::tuple<at::Tensor,at::Tensor> native_dropout_xpu", dropout_file)

        # clean tmporary file
        import shutil
        assert os.path.exists(install_dir)
        shutil.rmtree(install_dir)
        
        return True

if __name__ == "__main__":
    unittest.main()