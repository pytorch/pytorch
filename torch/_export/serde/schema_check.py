# mypy: allow-untyped-defs
import dataclasses
import hashlib
import re
import typing
from enum import IntEnum
from typing import Any, Dict, Optional, Union, List

from torch._export.serde import schema
from torch._export.serde.union import _Union


class SchemaUpdateError(Exception):
    pass


def _check(x, msg):
    if not x:
        raise SchemaUpdateError(msg)


def _staged_schema():
    ret: Dict[str, Any] = {}
    defs = {}
    cpp_type_defs: List[str] = []

    def _handle_aggregate(ty):
        def dump_type(t):
            if isinstance(t, type):
                return t.__name__, t.__name__
            elif isinstance(t, str):
                assert t in defs
                return t, t
            elif o := typing.get_origin(t):
                # Lemme know if there's a better way to do this.
                if o == list:
                    yaml_head, cpp_head = "List", "std::vector"
                elif o == dict:
                    yaml_head, cpp_head = "Dict", "std::unordered_map"
                elif o == tuple:
                    if typing.get_args(t) == ():
                        return "Tuple[()]", "std::tuple<>"
                    yaml_head, cpp_head = "Tuple", "std::tuple"
                elif o == Union:
                    args = typing.get_args(t)
                    assert len(args) == 2 and args[1] == type(None)
                    yaml_type, cpp_type = dump_type(args[0])
                    return f"Optional[{yaml_type}]", f"std::optional<{cpp_type}>"
                else:
                    raise AssertionError(f"Type {t} is not supported in export schema.")
                yaml_arg_types, cpp_arg_types = zip(*[dump_type(x) for x in typing.get_args(t)])
                return (
                    f"{yaml_head}[{', '.join(yaml_arg_types)}]"
                ), (
                    f"{cpp_head}<{', '.join(cpp_arg_types)}>"
                )
            elif t == ():
                return "()", ""
            else:
                raise AssertionError(f"Type {t} is not supported in export schema.")

        def dump_cpp_value(v):
            if v is None:
                return "std::nullopt"
            elif v == True:
                return "true"
            elif v == False:
                return "false"
            elif v == {}:
                return "{}"
            elif v == []:
                return "{}"
            elif v == ():
                return "{}"
            elif v == "":
                return '""'
            else:
                raise AssertionError(f"Default value {v} is not supported yet in export schema.")

        def dump_field(f):
            t, cpp = dump_type(f.type)
            ret = {"type": t, "cpp_type": cpp}

            value = dataclasses.MISSING
            if f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()

            if t.startswith("Optional[") and value is not None:
                raise AssertionError(
                    f"Optional field {ty.__name__}.{f.name} must have default value to be None."
                )

            if value is not dataclasses.MISSING:
                default = str(value)
                ret["default"] = default
                ret["cpp_default"] = _dump_cpp_value(value)
            return ret

        return {f.name: dump_field(f) for f in dataclasses.fields(ty)}

    def _handle_int_enum(name, ty):
        ret[name] = {"kind": "enum", "fields": {x.name: x.value for x in ty}}

    def _handle_struct(name, ty):
        ret[name] = {"kind": "struct", "fields": _handle_aggregate(ty)}
        cpp_type_defs.append(f"""
class {name} {{
 private:
  
 public:
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT({name}, {", ".join(f.name for f in dataclasses.fields(ty))});
}};
""")

    def _handle_union(name, ty):
        ret[name] = {"kind": "union", "fields": _handle_aggregate(ty)}

    for name in dir(schema):
        if name.startswith("_"):
            continue

        value = getattr(schema, name)

        if hasattr(value, "__module__") and value.__module__ != schema.__name__:
            continue

        defs[name] = value

    for name, value in defs.items():
        if isinstance(value, type):
            if issubclass(value, IntEnum):
                _handle_int_enum(name, value)
            elif dataclasses.is_dataclass(value):
                if issubclass(value, _Union):
                    _handle_union(name, value)
                else:
                    _handle_struct(name, value)
            else:
                raise AssertionError(f"Unknown schema type {name}: {value}")
        elif isinstance(value, (int, tuple)):
            assert name in ("SCHEMA_VERSION", "TREESPEC_VERSION")
        else:
            raise AssertionError(f"Unknown variable {name}: {value}")

    ret["SCHEMA_VERSION"] = list(defs["SCHEMA_VERSION"])
    assert all(x > 0 for x in ret["SCHEMA_VERSION"])
    ret["TREESPEC_VERSION"] = defs["TREESPEC_VERSION"]
    assert ret["TREESPEC_VERSION"] > 0

    cpp_header = f"""
#pragma once

#include <nlohmann/json.hpp>

namespace torch {{
namespace runtime {{
{"\n".join(cpp_type_defs)}
}} // namespace runtime
}} // namespace torch
"""

    return ret, cpp_header


def _diff_schema(dst, src):
    additions = {key: src[key] for key in src.keys() - dst.keys()}
    subtractions = {key: dst[key] for key in dst.keys() - src.keys()}

    common_keys = src.keys() & dst.keys()

    versions = {"SCHEMA_VERSION", "TREESPEC_VERSION"}
    common_keys -= versions

    for key in common_keys:
        src_kind = src[key]["kind"]
        src_fields = src[key]["fields"]
        dst_kind = dst[key]["kind"]
        dst_fields = dst[key]["fields"]
        _check(
            src_kind == dst_kind,
            f"Type {key} changed kind from {dst_kind} to {src_kind}",
        )
        assert isinstance(src_fields, dict) and isinstance(dst_fields, dict)
        added_fields = {
            key: src_fields[key] for key in src_fields.keys() - dst_fields.keys()
        }
        subtracted_fields = {
            key: dst_fields[key] for key in dst_fields.keys() - src_fields.keys()
        }
        common_fields = src_fields.keys() & dst_fields.keys()

        for field in common_fields:
            src_field = src_fields[field]
            dst_field = dst_fields[field]
            if src_kind == "struct":
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
                if "default" in src_field and "default" not in dst_field:
                    added_fields[field] = {}
                    added_fields[field]["default"] = src_field["default"]
                if "default" not in src_field and "default" in dst_field:
                    subtracted_fields[field] = {}
                    subtracted_fields[field]["default"] = dst_field["default"]
            elif src_kind == "enum":
                _check(
                    src_field == dst_field,
                    f"Value of the enum field {key}.{field} changed from {dst_field} to {src_field}",
                )
            elif src_kind == "union":
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
            else:
                raise AssertionError(f"Unknown kind {src_kind}: {key}")
        if len(added_fields) > 0:
            assert key not in additions
            additions[key] = {}
            additions[key]["fields"] = added_fields
        if len(subtracted_fields) > 0:
            assert key not in subtractions
            subtractions[key] = {}
            subtractions[key]["fields"] = subtracted_fields

    return additions, subtractions


def _hash_schema(s):
    return hashlib.sha256(repr(s).encode("utf-8")).hexdigest()


@dataclasses.dataclass
class _Commit:
    result: Dict[str, Any]
    checksum_result: str
    yaml_path: str
    additions: Dict[str, Any]
    subtractions: Dict[str, Any]
    base: Dict[str, Any]
    checksum_base: Optional[str]
    cpp_header: str
    cpp_header_path: str


def update_schema():
    import importlib.resources

    if importlib.resources.is_resource(__package__, "schema.yaml"):
        content = importlib.resources.read_text(__package__, "schema.yaml")
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", content)
        _check(match is not None, "checksum not found in schema.yaml")
        assert match is not None
        checksum_base = match.group(1)
        from yaml import load, Loader

        dst = load(content, Loader=Loader)
        assert isinstance(dst, dict)
    else:
        checksum_base = None
        dst = {"SCHEMA_VERSION": None, "TREESPEC_VERSION": None}

    src, cpp_header = _staged_schema()
    additions, subtractions = _diff_schema(dst, src)
    yaml_path = __package__.replace(".", "/") + "/schema.yaml"
    torch_prefix = "torch/"
    assert yaml_path.startswith(torch_prefix)  # sanity check

    return _Commit(
        result=src,
        checksum_result=_hash_schema(src),
        yaml_path=yaml_path,
        additions=additions,
        subtractions=subtractions,
        base=dst,
        checksum_base=checksum_base,
        cpp_header=cpp_header,
        cpp_header_path=torch_prefix + "csrc/utils/generated_serialization_types.h",
    )


def check(commit: _Commit, force_unsafe: bool = False):
    next_version = None
    reason = ""
    # Step 1: Detect major schema updates.
    if len(commit.additions) > 0:
        for k, v in commit.additions.items():
            if k not in commit.base:
                continue
            kind = commit.result[k]["kind"]
            fields = v["fields"]
            for f, d in fields.items():
                if "default" not in d and kind == "struct":
                    reason += (
                        f"Field {k}.{f} is added to schema.py without a default value as an incomparible change "
                        + "which requires major version bump.\n"
                    )
                    next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    if len(commit.subtractions) > 0:
        for k, v in commit.subtractions.items():
            if k not in commit.result:
                continue
            for f in v["fields"]:
                reason = f"Field {k}.{f} is removed from schema.py as an incompatible change which requires major version bump.\n"
            next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    if force_unsafe:
        reason += "--force-unsafe is used."
        next_version = commit.result["SCHEMA_VERSION"]
    else:
        # Step 2: Detect minor schema updates.
        if next_version is None and len(commit.additions) > 0:
            for k, v in commit.additions.items():
                for f in v["fields"]:
                    reason += (
                        f"Field {k}.{f} is added to schema.py as an compatible change "
                        + "which still requires minor version bump.\n"
                    )
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]
        if next_version is None and len(commit.subtractions) > 0:
            for k, v in commit.subtractions.items():
                for f in v["fields"]:
                    reason += (
                        f"Field {k}.{f} is removed from schema.py as an compatible change "
                        + "which still requires minor version bump.\n"
                    )
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]

    return next_version, reason
