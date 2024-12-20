# mypy: allow-untyped-defs
import dataclasses
import hashlib
import inspect
import re
import typing
from enum import IntEnum
from typing import Annotated, Any, Dict, ForwardRef, List, Optional, Tuple, Union

from torch._export.serde import schema
from torch._export.serde.union import _Union


class SchemaUpdateError(Exception):
    pass


def _check(x, msg):
    if not x:
        raise SchemaUpdateError(msg)


def _staged_schema():
    yaml_ret: Dict[str, Any] = {}
    defs = {}
    cpp_enum_defs: Dict[str, str] = {}
    cpp_class_defs: Dict[str, str] = {}
    cpp_type_decls: List[str] = []
    cpp_json_defs: List[str] = []
    thrift_enum_defs: List[str] = []
    thrift_type_defs: Dict[str, str] = {}

    def _handle_aggregate(ty) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        def dump_type(t, level: int) -> Tuple[str, str, str]:
            CPP_TYPE_MAP = {
                str: "std::string",
                int: "int64_t",
                float: "double",
                bool: "bool",
            }
            THRIFT_TYPE_MAP = {
                str: "string",
                int: "i64",
                float: "double",
                bool: "bool",
            }
            if isinstance(t, type):
                if t.__name__ in cpp_enum_defs:
                    return t.__name__, "int64_t", t.__name__
                else:
                    return (
                        t.__name__,
                        CPP_TYPE_MAP.get(t, t.__name__),
                        THRIFT_TYPE_MAP.get(t, t.__name__),
                    )
            elif isinstance(t, str):
                assert t in defs
                assert t not in cpp_enum_defs
                assert "[" not in t
                return t, f"ForwardRef<{t}>", t
            elif isinstance(t, ForwardRef):
                return (
                    t.__forward_arg__,
                    f"ForwardRef<{t.__forward_arg__}>",
                    t.__forward_arg__,
                )
            elif o := typing.get_origin(t):
                # Lemme know if there's a better way to do this.
                if o == list:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "List",
                        "std::vector",
                        "list<",
                        ">",
                    )
                elif o == dict:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "Dict",
                        "std::unordered_map",
                        "map<",
                        ">",
                    )
                elif o == Union:
                    assert level == 0, "Optional is only supported at the top level."
                    args = typing.get_args(t)
                    assert len(args) == 2 and args[1] == type(None)
                    yaml_type, cpp_type, thrift_type = dump_type(args[0], level + 1)
                    return (
                        f"Optional[{yaml_type}]",
                        f"std::optional<{cpp_type}>",
                        f"optional {thrift_type}",
                    )
                elif o == Annotated:
                    return dump_type(t.__origin__, level)
                else:
                    raise AssertionError(f"Type {t} is not supported in export schema.")
                yaml_arg_types, cpp_arg_types, thrift_arg_types = zip(
                    *[dump_type(x, level + 1) for x in typing.get_args(t)]
                )
                return (
                    (f"{yaml_head}[{', '.join(yaml_arg_types)}]"),
                    (f"{cpp_head}<{', '.join(cpp_arg_types)}>"),
                    f"{thrift_head}{', '.join(thrift_arg_types)}{thrift_tail}",
                )
            else:
                raise AssertionError(f"Type {t} is not supported in export schema.")

        def dump_cpp_value(v) -> str:
            if v is None:
                return "std::nullopt"
            elif v is True:
                return "true"
            elif v is False:
                return "false"
            elif v == {}:
                return "{}"
            elif v == []:
                return "{}"
            elif v == ():
                return "{}"
            elif isinstance(v, str):
                return f'"{v}"'
            else:
                raise AssertionError(
                    f"Default value {v} is not supported yet in export schema."
                )

        def dump_field(f) -> Tuple[Dict[str, Any], str, Optional[str], str, int]:
            t, cpp_type, thrift_type = dump_type(f.type, 0)
            ret = {"type": t}
            cpp_default: Optional[str] = None
            assert (
                typing.get_origin(f.type) == Annotated
            ), f"Field {f.name} must be annotated with an integer id."
            thrift_id = f.type.__metadata__[0]
            assert (
                type(thrift_id) is int
            ), f"Field {f.name} must be annotated with an integer id."

            value = dataclasses.MISSING
            if f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()

            if value is not dataclasses.MISSING:
                default = str(value)
                ret["default"] = default
                cpp_default = dump_cpp_value(value)

                if t.startswith("Optional[") and value is not None:
                    raise AssertionError(
                        f"Optional field {ty.__name__}.{f.name} must have default value to be None."
                    )

            return ret, cpp_type, cpp_default, thrift_type, thrift_id

        yaml_ret = {}
        cpp_ret = {}
        thrift_ret = {}
        thrift_ids = set()
        for f in dataclasses.fields(ty):
            yaml_res, cpp_type, cpp_default, thrift_type, thrift_id = dump_field(f)
            yaml_ret[f.name] = yaml_res
            cpp_ret[f.name] = {"cpp_type": cpp_type, "cpp_default": cpp_default}
            thrift_ret[f.name] = {"thrift_type": thrift_type, "thrift_id": thrift_id}
            if thrift_id in thrift_ids:
                raise AssertionError(
                    f"Duplicate thrift id {thrift_id} for field {f.name} in {ty.__name__}."
                )
            thrift_ids.add(thrift_id)
        return yaml_ret, cpp_ret, thrift_ret

    def _handle_int_enum(name, ty):
        yaml_ret[name] = {"kind": "enum", "fields": {x.name: x.value for x in ty}}
        cpp_enum_defs[
            name
        ] = f"""
enum class {name} {{
{chr(10).join([f"  {x.name} = {x.value}," for x in ty])}
}};
"""
        thrift_enum_defs.append(
            f"""
enum {name} {{
{chr(10).join([f"  {x.name} = {x.value}," for x in ty])}
}}
"""
        )

    def _handle_struct(name, ty):
        fields, cpp_fields, thrift_fields = _handle_aggregate(ty)
        yaml_ret[name] = {"kind": "struct", "fields": fields}
        field_decls = "\n".join(
            f"  {f['cpp_type']} {name}{' = ' + f['cpp_default'] if f['cpp_default'] is not None else ''};"
            for name, f in cpp_fields.items()
        )

        def accessor(name, ty):
            type_name = fields[name]["type"]
            if type_name in cpp_enum_defs:
                return f"""
  {type_name} get_{name}() const {{
    return static_cast<{type_name}>({name});
  }}
"""
            return f"""
  const {ty}& get_{name}() const {{
    return {name};
  }}
"""

        to_json_decl = f"void to_json(nlohmann::json& nlohmann_json_j, const {name}& nlohmann_json_t)"
        to_json_def = f"""{{
{chr(10).join([f'  nlohmann_json_j["{name}"] = nlohmann_json_t.{name};' for name, f in cpp_fields.items()])}
}}
"""
        from_json_decl = f"void from_json(const nlohmann::json& nlohmann_json_j, {name}& nlohmann_json_t)"

        from_json_def = f"""{{
  {name} nlohmann_json_default_obj;
{chr(10).join(
    [f'  nlohmann_json_t.{name} = nlohmann_json_j.value("{name}", nlohmann_json_default_obj.{name});'
    for name, f in cpp_fields.items()])}
}}
"""
        cpp_class_defs[
            name
        ] = f"""
class {name} {{
 private:
{field_decls}

 public:
{"".join([accessor(name, f["cpp_type"]) for name, f in cpp_fields.items()])}
  friend {to_json_decl};
  friend {from_json_decl};
}};
"""
        cpp_json_defs.append(f"inline {to_json_decl} {to_json_def}")
        cpp_json_defs.append(f"inline {from_json_decl} {from_json_def}")
        cpp_type_decls.append(f"class {name};")

        thrift_type_defs[
            name
        ] = f"""
struct {name} {{
{chr(10).join(f"  {f['thrift_id']}: {f['thrift_type']} {n};" for n, f in thrift_fields.items())}
}}"""

    def _handle_union(name, ty):
        fields, cpp_fields, thrift_fields = _handle_aggregate(ty)
        yaml_ret[name] = {"kind": "union", "fields": fields}

        def accessor(name, ty, idx):
            return f"""
  const {ty}& get_{name}() const {{
    return std::get<{idx + 1}>(variant_);
  }}
"""

        to_json_branches = "".join(
            [
                f"""
    if (nlohmann_json_t.tag_ == Tag::{name.upper()}) {{
      nlohmann_json_j["{name}"] = nlohmann_json_t.get_{name}();
      return;
    }}"""
                for idx, (name, f) in enumerate(cpp_fields.items())
            ]
        )
        from_json_branches = "".join(
            [
                f"""
    if (nlohmann_json_j.contains("{name}")) {{
      nlohmann_json_t.variant_.emplace<{idx + 1}>(nlohmann_json_j.at("{name}").template get<{f["cpp_type"]}>());
      nlohmann_json_t.tag_ = Tag::{name.upper()};
      return;
    }}"""
                for idx, (name, f) in enumerate(cpp_fields.items())
            ]
        )

        cpp_class_defs[
            name
        ] = f"""
class {name} {{
  struct Void {{}};

 public:
  enum class Tag {{
    {", ".join([name.upper() for name in cpp_fields])}
  }};

 private:
  std::variant<Void, {", ".join(f["cpp_type"] for f in cpp_fields.values())}> variant_;
  Tag tag_;

 public:
  Tag tag() const {{
    return tag_;
  }}
{"".join([accessor(name, f["cpp_type"], idx) for idx, (name, f) in enumerate(cpp_fields.items())])}
  friend void to_json(nlohmann::json& nlohmann_json_j, const {name}& nlohmann_json_t) {{
{to_json_branches}
  }}

  friend void from_json(const nlohmann::json& nlohmann_json_j, {name}& nlohmann_json_t) {{
{from_json_branches}
  }}
}};
"""
        cpp_type_decls.append(f"class {name};")

        thrift_type_defs[
            name
        ] = f"""
union {name} {{
{chr(10).join(f"  {f['thrift_id']}: {f['thrift_type']} {n};" for n, f in thrift_fields.items())}
}}"""

    for name in dir(schema):
        if name.startswith("_"):
            continue

        value = getattr(schema, name)

        if hasattr(value, "__module__") and value.__module__ != schema.__name__:
            continue

        defs[name] = value

    class_ordering = {}
    for name, value in defs.items():
        if isinstance(value, type):
            if issubclass(value, IntEnum):
                _handle_int_enum(name, value)
            elif dataclasses.is_dataclass(value):
                class_ordering[name] = inspect.findsource(value)[1]
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

    yaml_ret["SCHEMA_VERSION"] = list(defs["SCHEMA_VERSION"])
    assert all(x > 0 for x in yaml_ret["SCHEMA_VERSION"])
    yaml_ret["TREESPEC_VERSION"] = defs["TREESPEC_VERSION"]
    assert yaml_ret["TREESPEC_VERSION"] > 0

    cpp_header = f"""
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef NLOHMANN_JSON_NAMESPACE_BEGIN
#define NLOHMANN_JSON_NAMESPACE_BEGIN namespace nlohmann {{
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_END
#define NLOHMANN_JSON_NAMESPACE_END }}
#endif

// https://github.com/nlohmann/json/pull/2117
NLOHMANN_JSON_NAMESPACE_BEGIN
template <typename T>
struct adl_serializer<std::optional<T>> {{
  static void to_json(json& j, const std::optional<T>& opt) {{
    if (opt == std::nullopt) {{
      j = nullptr;
    }} else {{
      j = *opt; // this will call adl_serializer<T>::to_json which will
                // find the free function to_json in T's namespace!
    }}
  }}

  static void from_json(const json& j, std::optional<T>& opt) {{
    if (j.is_null()) {{
      opt = std::nullopt;
    }} else {{
      opt = j.template get<T>(); // same as above, but with
                                 // adl_serializer<T>::from_json
    }}
  }}
}};
NLOHMANN_JSON_NAMESPACE_END

namespace torch {{
namespace _export {{

template <typename T>
class ForwardRef {{
  static_assert(!std::is_reference_v<T>, "ForwardRef cannot be a reference type");

 public:
  ForwardRef(): ptr_(std::make_unique<T>()) {{}}
  ForwardRef(ForwardRef<T>&&) = default;
  ForwardRef(const ForwardRef<T>& other): ptr_(std::make_unique<T>(*other.ptr_)) {{}}
  ForwardRef<T>& operator=(ForwardRef<T>&&) = default;
  ForwardRef<T>& operator=(const ForwardRef<T>& other) {{
    ptr_ = std::make_unique<T>(*other.ptr_);
  }}
  const T& operator*() const {{
    return *ptr_;
  }}

  const T* operator->() const {{
    return ptr_.get();
  }}

  void emplace(T&& t) {{
    ptr_ = std::make_unique<T>(std::move(t));
  }}

 private:
  std::unique_ptr<T> ptr_;
}};

template <typename T>
void to_json(nlohmann::json& j, const ForwardRef<T>& p) {{
  j = *p;
}}

template <typename T>
void from_json(const nlohmann::json& j, ForwardRef<T>& p) {{
  p.emplace(j.template get<T>());
}}

{chr(10).join(cpp_type_decls)}
{"".join(cpp_enum_defs.values())}
{"".join(dict(sorted(cpp_class_defs.items(), key=lambda x: class_ordering[x[0]])).values())}
{chr(10).join(cpp_json_defs)}
}} // namespace _export
}} // namespace torch
"""
    thrift_schema = f"""
namespace py3 torch._export
namespace cpp2 torch._export.schema
{chr(10).join(thrift_enum_defs)}
{chr(10).join(dict(sorted(thrift_type_defs.items(), key=lambda x: class_ordering[x[0]])).values())}
"""
    return yaml_ret, cpp_header, thrift_schema


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


def _hash_content(s: str):
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


@dataclasses.dataclass
class _Commit:
    result: Dict[str, Any]
    checksum_next: str
    yaml_path: str
    additions: Dict[str, Any]
    subtractions: Dict[str, Any]
    base: Dict[str, Any]
    checksum_head: Optional[str]
    cpp_header: str
    cpp_header_path: str
    thrift_checksum_head: Optional[str]
    thrift_checksum_real: Optional[str]
    thrift_checksum_next: str
    thrift_schema: str
    thrift_schema_path: str


def update_schema():
    import importlib.resources

    if importlib.resources.is_resource(__package__, "schema.yaml"):
        content = importlib.resources.read_text(__package__, "schema.yaml")
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", content)
        _check(match is not None, "checksum not found in schema.yaml")
        assert match is not None
        checksum_head = match.group(1)

        thrift_content = importlib.resources.read_text(
            __package__, "export_schema.thrift"
        )
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", thrift_content)
        _check(match is not None, "checksum not found in export_schema.thrift")
        assert match is not None
        thrift_checksum_head = match.group(1)
        thrift_content = thrift_content.splitlines()
        assert thrift_content[0].startswith("// @" + "generated")
        assert thrift_content[1].startswith("// checksum<<")
        thrift_checksum_real = _hash_content("\n".join(thrift_content[2:]))

        from yaml import load, Loader

        dst = load(content, Loader=Loader)
        assert isinstance(dst, dict)
    else:
        checksum_head = None
        thrift_checksum_head = None
        thrift_checksum_real = None
        dst = {"SCHEMA_VERSION": None, "TREESPEC_VERSION": None}

    src, cpp_header, thrift_schema = _staged_schema()
    additions, subtractions = _diff_schema(dst, src)
    yaml_path = __package__.replace(".", "/") + "/schema.yaml"
    thrift_schema_path = __package__.replace(".", "/") + "/export_schema.thrift"
    torch_prefix = "torch/"
    assert yaml_path.startswith(torch_prefix)  # sanity check
    assert thrift_schema_path.startswith(torch_prefix)  # sanity check

    return _Commit(
        result=src,
        checksum_next=_hash_content(repr(src)),
        yaml_path=yaml_path,
        additions=additions,
        subtractions=subtractions,
        base=dst,
        checksum_head=checksum_head,
        cpp_header=cpp_header,
        cpp_header_path=torch_prefix + "csrc/utils/generated_serialization_types.h",
        thrift_checksum_head=thrift_checksum_head,
        thrift_checksum_real=thrift_checksum_real,
        thrift_checksum_next=_hash_content(thrift_schema),
        thrift_schema=thrift_schema,
        thrift_schema_path=thrift_schema_path,
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
                if kind == "struct" and "default" not in d:
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
