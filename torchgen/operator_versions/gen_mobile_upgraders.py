#!/usr/bin/env python3
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode

from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
    MOBILE_UPGRADERS_HEADER_DESCRIPTION,
)


class ByteCode(Enum):
    instructions = 1
    constants = 2
    types = 3
    operators = 4
    register_size = 5


EXCLUDED_OP_SET = [
    "aten::full.names",
    "aten::full.out",
    "aten::full",
]

EXCLUE_UPGRADER_SET = ["full_0_4", "full_out_0_4"]

ONE_INSTRUCTION = CodeTemplate(
    """
    Instruction{OpCode::${operator_name}, ${X}, ${N}},"""
)

INSTRUCTION_LIST = CodeTemplate(
    """std::vector<Instruction>({
        ${instruction_list}
    }), // instructions list"""
)

ONE_CONSTANT = CodeTemplate(
    """
    c10::IValue(${constant}),"""
)

CONSTANT_LIST = CodeTemplate(
    """std::vector<c10::IValue>({
        ${constant_list}
    }), // constants list"""
)

CONSTANTS_LIST_EMPTY = """std::vector<c10::IValue>(), // constants list"""

ONE_TYPE = CodeTemplate("""c10::parseType("${type_str}"),""")

TYPE_LIST = CodeTemplate(
    """std::vector<c10::TypePtr>({
        ${type_list}
    }), // types list"""
)

TYPE_LIST_EMPTY = """std::vector<c10::TypePtr>(), // types list"""

ONE_OPERATOTR_STRING = CodeTemplate(
    """
    OperatorString({"${operator_name}", "${overload_name}", ${num_of_args}}),"""
)

OPERATOR_STRING_LIST = CodeTemplate(
    """
    std::vector<OperatorString>({
        ${operator_string_list}
    }), // operators list"""
)

ONE_UPGRADER_FUNCTION = CodeTemplate(
    """
    mobile::Function::registerFunc(
        "${upgrader_name}",
        ${instruction_list},
        ${constant_list},
        ${type_list},
        ${register_size}
    )"""
)

ONE_UPGRADER_SRC = CodeTemplate(
    """
    ByteCodeFunctionWithOperator({
        ${bytecode_function},
        ${operator_string_list}
    }),"""
)


ONE_UPGRADER_IN_VERSION_MAP = CodeTemplate(
    """Upgrader({${upgrader_min_version}, ${upgrader_max_version}, "${upgrader_name}", ${bytecode_func_index}})"""
)  # noqa: E501

ONE_OPERATOR_IN_VERSION_MAP = CodeTemplate(
    """
    {std::string("${operator_name}"),
        std::vector<Upgrader>({
            ${upgrader_list_in_version_map}
        })},"""
)


OPERATOR_VERSION_MAP = CodeTemplate(
    """
const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile() {
  static std::unordered_map<std::string, std::vector<Upgrader>>
        operatorVersionMapForMobile({
            ${operator_list_in_version_map}
      });
  return operatorVersionMapForMobile;
}
"""
)


UPGRADER_CPP_SRC = CodeTemplate(
    MOBILE_UPGRADERS_HEADER_DESCRIPTION
    + """
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

// clang-format off

// From operator_versions_map
${operator_version_map}

const std::vector<ByteCodeFunctionWithOperator>& getUpgraderBytecodeList() {
  auto generate_upgrader_bytecode_list = []() {
    std::vector<ByteCodeFunctionWithOperator> upgrader_function_list({
               ${upgrader_bytecode}
            });
    for (const auto& upgrader_function : upgrader_function_list) {
      for (const auto& op : upgrader_function.operators) {
        upgrader_function.function.append_operator(
            op.name,
            op.overload_name,
            op.num_specified_args);
      }
    }
    return upgrader_function_list;
  };
  static std::vector<ByteCodeFunctionWithOperator> upgraderBytecodeList =
      generate_upgrader_bytecode_list();
  return upgraderBytecodeList;
}

// clang-format on

} // namespace jit
} // namespace torch
"""
)

UPGRADER_MOBILE_FILE_NAME = "upgrader_mobile.cpp"

UPGRADER_ELEMENT = CodeTemplate(
    """\
Upgrader({${min_version}, ${max_version}, ${operator_name}, ${index}}),
"""
)

PER_OPERATOR_UPGRADER_LIST = CodeTemplate(
    """\
{
  std::string(${operator_name}),
  std::vector<Upgrader>({${upgrader_list}});
}
"""
)


def construct_instruction(instruction_list_from_yaml: List[Any]) -> str:
    instruction_list_part = []
    for instruction in instruction_list_from_yaml:
        instruction_list_part.append(
            ONE_INSTRUCTION.substitute(
                operator_name=instruction[0],
                X=instruction[1],
                N=instruction[2],
            )
        )
    return INSTRUCTION_LIST.substitute(
        instruction_list="".join(instruction_list_part).lstrip("\n")
    )


def construct_constants(constants_list_from_yaml: List[Any]) -> str:
    constants_list_part = []
    for constant_from_yaml in constants_list_from_yaml:
        convert_constant = None
        if isinstance(constant_from_yaml, str):
            # Add quotes if it's string
            convert_constant = f'"{constant_from_yaml}"'
        elif isinstance(constant_from_yaml, bool):
            convert_constant = "true" if constant_from_yaml else "false"
        elif constant_from_yaml is None:
            convert_constant = ""
        elif isinstance(constant_from_yaml, int):
            convert_constant = str(constant_from_yaml)
        else:
            raise ValueError(
                f"The type of {constant_from_yaml} is {type(constant_from_yaml)}. "
                "Please add change in construct_constants function in gen_mobile_upgraders.py."
            )
        constants_list_part.append(ONE_CONSTANT.substitute(constant=convert_constant))
    if len(constants_list_part) == 0:
        return CONSTANTS_LIST_EMPTY
    return CONSTANT_LIST.substitute(
        constant_list="".join(constants_list_part).lstrip("\n")
    )


def construct_operators(operator_list_from_yaml: List[Any]) -> str:
    operator_list_part = []
    for operator in operator_list_from_yaml:
        operator_list_part.append(
            ONE_OPERATOTR_STRING.substitute(
                operator_name=operator[0],
                overload_name=operator[1],
                num_of_args=operator[2],
            )
        )
    return OPERATOR_STRING_LIST.substitute(
        operator_string_list="".join(operator_list_part).lstrip("\n")
    )


def construct_types(types_tr_list_from_yaml: List[Any]) -> str:
    types_tr_list_part = []
    for types_tr in types_tr_list_from_yaml:
        types_tr_list_part.append(ONE_TYPE.substitute(type_str=types_tr))
    if len(types_tr_list_part) == 0:
        return TYPE_LIST_EMPTY
    return TYPE_LIST.substitute(type_list="".join(types_tr_list_part).lstrip("\n"))


def construct_register_size(register_size_from_yaml: int) -> str:
    if not isinstance(register_size_from_yaml, int):
        raise ValueError(
            f"Input register size is {register_size_from_yaml} and"
            "it's type is {type(register_size_from_yaml)}. An int type is expected."
        )
    return str(register_size_from_yaml)


def construct_version_maps(
    upgrader_bytecode_function_to_index_map: Dict[str, Any]
) -> str:
    version_map = torch._C._get_operator_version_map()
    sorted_version_map_ = sorted(version_map.items(), key=lambda item: item[0])  # type: ignore[no-any-return]
    sorted_version_map = dict(sorted_version_map_)

    operator_list_in_version_map_part = []
    for op_name in sorted_version_map:
        upgraders_in_version_map_part = []
        # TODO: remove the skip after these two operators schemas are fixed
        if op_name in EXCLUDED_OP_SET:
            continue
        upgrader_ranges = torch._C._get_upgrader_ranges(op_name)
        upgrader_entries = sorted_version_map[op_name]
        assert len(upgrader_ranges) == len(upgrader_entries)
        for idx, upgrader_entry in enumerate(upgrader_entries):
            upgrader_name = upgrader_entry.upgrader_name
            bytecode_function_index = upgrader_bytecode_function_to_index_map[
                upgrader_name
            ]
            upgraders_in_version_map_part.append(
                ONE_UPGRADER_IN_VERSION_MAP.substitute(
                    upgrader_min_version=upgrader_ranges[idx].min_version,
                    upgrader_max_version=upgrader_ranges[idx].max_version,
                    upgrader_name=upgrader_name,
                    bytecode_func_index=bytecode_function_index,
                )
            )
        operator_list_in_version_map_part.append(
            ONE_OPERATOR_IN_VERSION_MAP.substitute(
                operator_name=op_name,
                upgrader_list_in_version_map="".join(upgraders_in_version_map_part),
            )
        )
    return OPERATOR_VERSION_MAP.substitute(
        operator_list_in_version_map="".join(operator_list_in_version_map_part).lstrip(
            "\n"
        )
    )


def get_upgrader_bytecode_function_to_index_map(
    upgrader_dict: List[Dict[str, Any]]
) -> Dict[str, Any]:
    upgrader_bytecode_function_to_index_map = {}
    index = 0
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name, bytecode in upgrader_bytecode.items():
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            upgrader_bytecode_function_to_index_map[upgrader_name] = index
            index += 1
    return upgrader_bytecode_function_to_index_map


def write_cpp(cpp_path: str, upgrader_dict: List[Dict[str, Any]]) -> None:
    body_parts = []
    upgrader_bytecode_function_to_index_map = (
        get_upgrader_bytecode_function_to_index_map(upgrader_dict)
    )
    version_map_src = construct_version_maps(upgrader_bytecode_function_to_index_map)
    all_upgrader_src_string = []
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name, bytecode in upgrader_bytecode.items():
            # TODO: remove the skip after these two operators schemas are fixed
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            instruction_list_str = ""
            constant_list_str = ""
            type_list_str = ""
            register_size_str = ""
            operator_list_str = ""
            for table_name, contents in bytecode.items():
                element = ByteCode[table_name]
                body_string = ""
                if element is ByteCode.instructions:
                    instruction_list_str = construct_instruction(contents)
                elif element is ByteCode.constants:
                    constant_list_str = construct_constants(contents)
                elif element is ByteCode.operators:
                    operator_list_str = construct_operators(contents)
                elif element is ByteCode.types:
                    type_list_str = construct_types(contents)
                elif element is ByteCode.register_size:
                    register_size_str = construct_register_size(contents)

            one_upgrader_function_string = ONE_UPGRADER_FUNCTION.substitute(
                upgrader_name=upgrader_name,
                instruction_list=instruction_list_str,
                constant_list=constant_list_str,
                type_list=type_list_str,
                register_size=register_size_str,
            )
            one_upgrader_src_string = ONE_UPGRADER_SRC.substitute(
                bytecode_function=one_upgrader_function_string.lstrip("\n"),
                operator_string_list=operator_list_str.lstrip("\n"),
            )
            all_upgrader_src_string.append(one_upgrader_src_string)

    upgrader_file_content = UPGRADER_CPP_SRC.substitute(
        operator_version_map=version_map_src,
        upgrader_bytecode="".join(all_upgrader_src_string).lstrip("\n"),
    )
    body_parts.append(upgrader_file_content)
    print("writing file to : ", cpp_path + "/" + UPGRADER_MOBILE_FILE_NAME)
    with open(os.path.join(cpp_path, UPGRADER_MOBILE_FILE_NAME), "wb") as out_file:
        final_output = "".join(body_parts)
        out_file.write(upgrader_file_content.encode("utf-8"))


def sort_upgrader(upgrader_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_upgrader_list = sorted(
        upgrader_list, key=lambda one_upgrader: next(iter(one_upgrader))
    )
    return sorted_upgrader_list


def main() -> None:
    upgrader_list = generate_upgraders_bytecode()
    sorted_upgrader_list = sort_upgrader(upgrader_list)
    for up in sorted_upgrader_list:
        print("after sort upgrader : ", next(iter(up)))

    pytorch_dir = Path(__file__).resolve().parents[2]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "mobile"
    write_cpp(str(upgrader_path), sorted_upgrader_list)


if __name__ == "__main__":
    main()
