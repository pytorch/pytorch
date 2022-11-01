import copy
import os

import yaml

from torchgen.code_template import CodeTemplate
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader # type: ignore[misc]

# https://gist.github.com/pypt/94d747fe5180851196eb
class UniqueKeyLoader(Loader):
    def construct_mapping(self, node, deep=False): # type: ignore[no-untyped-def]
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep) # type: ignore[no-untyped-call]
            try:
                hash(key)
            except TypeError:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key ",
                    key_node.start_mark,
                )
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                )
            value = self.construct_object(value_node, deep=deep) # type: ignore[no-untyped-call]
            mapping[key] = value
        return mapping


class GLSLGenerator(object):
    standard_header = """
#version 450 core
#define PRECISION $precision
#define FORMAT $format

"""

    def __init__(self): # type: ignore[no-untyped-def]
        self.ops_template_params = {}

    def add_params_yaml(self, parameters_yaml_file) -> None:
        all_template_params = {}
        with open(parameters_yaml_file, "r") as f:
            contents = yaml.load(f, Loader=UniqueKeyLoader)
            for key in contents:
                all_template_params[key] = contents[key]
        self.validate_and_construct_op_params(all_template_params) # type: ignore[no-untyped-call]

    def validate_and_construct_op_params(self, all_template_params) -> None:
        for op in all_template_params:
            if op in self.ops_template_params:
                raise KeyError(f"{op} params file has already been parsed")
            op_params_default_vals = all_template_params[op][
                "parameter_names_with_default_values"
            ]
            template_params_set = set(op_params_default_vals.keys())
            self.ops_template_params[op] = []
            self.ops_template_params[op].append(op_params_default_vals)
            op_template_params_values = all_template_params[op]["parameter_values"]
            for param_vals in op_template_params_values:
                param_vals_set = set(param_vals.keys())
                missing_keys = template_params_set - param_vals_set
                invalid_keys = param_vals_set - template_params_set
                if (len(invalid_keys)) > 0:
                    raise KeyError(f"Invalid keys {invalid_keys} are found")
                param_vals_copy = copy.deepcopy(param_vals)
                for key in missing_keys:
                    param_vals_copy[key] = op_params_default_vals[key]
                self.ops_template_params[op].append(param_vals_copy)

    def generate(self, glsl_template_in, out_dir) -> None:
        glsl_template_name = os.path.basename(glsl_template_in)
        op_name, extension_name = glsl_template_name.split(".")
        if extension_name != "glslt":
            raise TypeError(f"invalid file type for glsl template {extension_name}")
        if op_name not in self.ops_template_params:
            raise KeyError(f"{op_name} params have not been populated")
        code_template = CodeTemplate.from_file(glsl_template_in)
        for template_params in self.ops_template_params[op_name]:
            content = GLSLGenerator.standard_header
            param_vals_string = "x".join([str(i) for i in template_params.values()])
            output_file_name = op_name + "_" + param_vals_string + ".glsl"
            content += code_template.substitute(template_params)
            output_file = os.path.join(out_dir, output_file_name)
            with open(output_file, "w") as f:
                f.write(content)


# Remove this
if __name__ == "__main__":
    pass
