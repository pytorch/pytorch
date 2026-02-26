# mypy: allow-untyped-defs
import dataclasses
import hashlib
import inspect
import re
import types
import typing
from enum import IntEnum
from typing import Annotated, Any, ForwardRef, Union

from torch._export.serde import schema
from torch._export.serde.union import _Union


class SchemaUpdateError(Exception):
    pass


def _check(x, msg):
    if not x:
        raise SchemaUpdateError(msg)


_CPP_TYPE_MAP = {
    str: "std::string",
    int: "int64_t",
    float: "F64",
    bool: "bool",
}

_THRIFT_TYPE_MAP = {
    str: "string",
    int: "i64",
    float: "double",
    bool: "bool",
}


def _staged_schema():
    yaml_ret: dict[str, Any] = {}
    defs = {}
    cpp_enum_defs: dict[str, str] = {}
    cpp_class_defs: dict[str, str] = {}
    cpp_type_decls: list[str] = []
    cpp_json_defs: list[str] = []
    thrift_enum_defs: list[str] = []
    thrift_type_defs: dict[str, str] = {}

    def _handle_aggregate(ty) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        def dump_type(t, level: int) -> tuple[str, str, str]:
            if getattr(t, "__name__", None) in cpp_enum_defs:
                return t.__name__, "int64_t", t.__name__
            elif t in _CPP_TYPE_MAP:
                return (t.__name__, _CPP_TYPE_MAP[t], _THRIFT_TYPE_MAP[t])
            elif isinstance(t, str):
                if t not in defs:
                    raise AssertionError(f"type {t} not in defs")
                if t in cpp_enum_defs:
                    raise AssertionError(f"type {t} unexpectedly in cpp_enum_defs")
                if "[" in t:
                    raise AssertionError(f"type {t} contains '[' which is not allowed")
                return t, f"ForwardRef<{t}>", t
            elif isinstance(t, ForwardRef):
                return (
                    t.__forward_arg__,
                    f"ForwardRef<{t.__forward_arg__}>",
                    t.__forward_arg__,
                )
            elif o := typing.get_origin(t):
                # Lemme know if there's a better way to do this.
                if o is list:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "List",
                        "std::vector",
                        "list<",
                        ">",
                    )
                elif o is dict:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "Dict",
                        "std::unordered_map",
                        "map<",
                        ">",
                    )
                elif o is Union or o is types.UnionType:
                    if level != 0:
                        raise AssertionError(
                            f"Optional is only supported at the top level, got level={level}"
                        )
                    args = typing.get_args(t)
                    if len(args) != 2 or args[1] is not type(None):
                        raise AssertionError(
                            f"expected Optional type with 2 args ending in None, got {args}"
                        )
                    yaml_type, cpp_type, thrift_type = dump_type(args[0], level + 1)
                    return (
                        f"Optional[{yaml_type}]",
                        f"std::optional<{cpp_type}>",
                        f"optional {thrift_type}",
                    )
                elif o is Annotated:
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
            elif isinstance(t, type):
                return (t.__name__, t.__name__, t.__name__)
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

        def dump_field(f) -> tuple[dict[str, Any], str, str | None, str, int]:
            t, cpp_type, thrift_type = dump_type(f.type, 0)
            ret = {"type": t}
            cpp_default: str | None = None
            if typing.get_origin(f.type) is not Annotated:
                raise AssertionError(
                    f"Field {f.name} must be annotated with an integer id."
                )
            thrift_id = f.type.__metadata__[0]
            if type(thrift_id) is not int:
                raise AssertionError(
                    f"Field {f.name} must be annotated with an integer id, got {type(thrift_id)}"
                )

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
        cpp_enum_defs[name] = f"""
enum class {name} {{
{chr(10).join([f"  {x.name} = {x.value}," for x in ty])}
}};

inline std::string_view printEnum(const {name}& e) {{
  switch (e) {{
{chr(10).join([f"    case {name}::{x.name}: return {chr(34)}{x.name}{chr(34)};" for x in ty])}
    default:
      throw std::runtime_error("Unknown enum value");
  }}
}}

inline void parseEnum(std::string_view s, {name}& t) {{
{chr(10).join([f"  if (s == {chr(34)}{x.name}{chr(34)}) {{ t = {name}::{x.name}; return; }}" for x in ty])}
  throw std::runtime_error("Unknown enum value: " + std::string{{s}});
}}
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

  void set_{name}({type_name} def) {{
    {name} = static_cast<int64_t>(def);
  }}
"""
            return f"""
  const {ty}& get_{name}() const {{
    return {name};
  }}

  void set_{name}({ty} def) {{
    {name} = std::move(def);
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
{
            chr(10).join(
                [
                    f'  nlohmann_json_t.{name} = nlohmann_json_j.value("{name}", nlohmann_json_default_obj.{name});'
                    for name, f in cpp_fields.items()
                ]
            )
        }
}}
"""
        cpp_class_defs[name] = f"""
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

        thrift_type_defs[name] = f"""
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

  void set_{name}({ty} def) {{
    variant_.emplace<{idx + 1}>(std::move(def));
    tag_ = Tag::{name.upper()};
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

        cpp_class_defs[name] = f"""
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

inline std::string_view printEnum(const {name}::Tag& e) {{
  switch (e) {{
{chr(10).join([f"    case {name}::Tag::{x.upper()}: return {chr(34)}{x.upper()}{chr(34)};" for x in cpp_fields])}
    default:
      throw std::runtime_error("Unknown enum value");
  }}
}}

inline void parseEnum(std::string_view s, {name}::Tag& t) {{
{chr(10).join([f"  if (s == {chr(34)}{x.upper()}{chr(34)}) {{ t = {name}::Tag::{x.upper()}; return; }}" for x in cpp_fields])}
  throw std::runtime_error("Unknown enum value: " + std::string{{s}});
}}

"""
        cpp_type_decls.append(f"class {name};")

        thrift_type_defs[name] = f"""
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
            if name not in ("SCHEMA_VERSION", "TREESPEC_VERSION"):
                raise AssertionError(
                    f"expected SCHEMA_VERSION or TREESPEC_VERSION, got {name}"
                )
        elif isinstance(value, dict):
            # Skip mapping dictionaries used for codegen
            pass
        else:
            raise AssertionError(f"Unknown variable {name}: {value}")

    yaml_ret["SCHEMA_VERSION"] = list(defs["SCHEMA_VERSION"])
    if not all(x > 0 for x in yaml_ret["SCHEMA_VERSION"]):
        raise AssertionError(
            f"all SCHEMA_VERSION values must be > 0, got {yaml_ret['SCHEMA_VERSION']}"
        )
    yaml_ret["TREESPEC_VERSION"] = defs["TREESPEC_VERSION"]
    if yaml_ret["TREESPEC_VERSION"] <= 0:
        raise AssertionError(
            f"TREESPEC_VERSION must be > 0, got {yaml_ret['TREESPEC_VERSION']}"
        )

    cpp_header = f"""
#pragma once

#include <optional>
#include <stdexcept>
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
  ForwardRef(ForwardRef<T>&&);
  ForwardRef(const ForwardRef<T>& other): ptr_(std::make_unique<T>(*other.ptr_)) {{}}
  ForwardRef<T>& operator=(ForwardRef<T>&&);
  ForwardRef<T>& operator=(const ForwardRef<T>& other) {{
    ptr_ = std::make_unique<T>(*other.ptr_);
    return *this;
  }}
  ~ForwardRef();
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

class F64 {{
 public:
  double get() const {{
    return value_;
  }}

  void set(double value) {{
    value_ = value;
  }}

 private:
  double value_;
}};

inline void to_json(nlohmann::json& j, const F64& f) {{
  if (std::isinf(f.get())) {{
    j = "Infinity";
  }} else if (std::isinf(-f.get())) {{
    j = "-Infinity";
  }} else if (std::isnan(f.get())) {{
    j = "NaN";
  }} else {{
    j = f.get();
  }}
}}

inline void from_json(const nlohmann::json& j, F64& f) {{
  if (j == "Infinity") {{
    f.set(std::numeric_limits<double>::infinity());
  }} else if (j == "-Infinity") {{
    f.set(-std::numeric_limits<double>::infinity());
  }} else if (j == "NaN") {{
    f.set(std::numeric_limits<double>::quiet_NaN());
  }} else {{
    f.set(j.get<double>());
  }}
}}

{chr(10).join(cpp_type_decls)}
{"".join(cpp_enum_defs.values())}
{"".join(dict(sorted(cpp_class_defs.items(), key=lambda x: class_ordering[x[0]])).values())}
{chr(10).join(cpp_json_defs)}

template <typename T> ForwardRef<T>::ForwardRef(ForwardRef<T>&&) = default;
template <typename T> ForwardRef<T>& ForwardRef<T>::operator=(ForwardRef<T>&&) = default;
template <typename T> ForwardRef<T>::~ForwardRef() = default;
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
        if not isinstance(src_fields, dict) or not isinstance(dst_fields, dict):
            raise AssertionError(
                f"expected dict fields, got src={type(src_fields)}, dst={type(dst_fields)}"
            )
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
            if key in additions:
                raise AssertionError(f"key {key} already in additions")
            additions[key] = {}
            additions[key]["fields"] = added_fields
        if len(subtracted_fields) > 0:
            if key in subtractions:
                raise AssertionError(f"key {key} already in subtractions")
            subtractions[key] = {}
            subtractions[key]["fields"] = subtracted_fields

    return additions, subtractions


def _hash_content(s: str):
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


def _generate_enum_converters() -> str:
    """Generate C++ converter functions from serialized enum values to c10 enums."""

    def validate_mapping(
        enum_class: type[IntEnum],
        mapping: dict[int, str],
        enum_name: str,
        skip_values: set[int],
    ) -> None:
        """Validate that all enum values have corresponding c10 mappings."""
        for member in enum_class:
            if member.value in skip_values:
                continue
            if member.value not in mapping:
                raise SchemaUpdateError(
                    f"{enum_name}.{member.name} (value={member.value}) is missing "
                    f"from {enum_name.upper()}_TO_C10 mapping in schema.py. "
                    f"Please add the mapping to the c10 enum name."
                )

    # Validate that all enum values have mappings (except UNKNOWN values)
    validate_mapping(
        schema.ScalarType,
        schema.SCALAR_TYPE_TO_C10,
        "ScalarType",
        {schema.ScalarType.UNKNOWN},
    )
    validate_mapping(
        schema.Layout,
        schema.LAYOUT_TO_C10,
        "Layout",
        {schema.Layout.Unknown},
    )
    validate_mapping(
        schema.MemoryFormat,
        schema.MEMORY_FORMAT_TO_C10,
        "MemoryFormat",
        {schema.MemoryFormat.Unknown},
    )

    def generate_converter(
        name: str,
        c10_type: str,
        mapping: dict[int, str],
        max_value: int,
    ) -> str:
        lines: list[str] = []
        for i in range(max_value + 1):
            if i in mapping:
                lines.append(
                    f"      static_cast<int>(c10::{c10_type}::{mapping[i]}), // {i}"
                )
            else:
                lines.append(f"      kInvalid, // {i}")

        return f"""
inline c10::{c10_type} convertSerialized{name}(int serialized_value) {{
  constexpr int kInvalid = -1;
  constexpr int k{name}Map[] = {{
{chr(10).join(lines)}
  }};
  constexpr int kMapSize = sizeof(k{name}Map) / sizeof(k{name}Map[0]);

  TORCH_CHECK(
      serialized_value >= 0 && serialized_value < kMapSize,
      "Serialized {name} value out of range: ",
      serialized_value);
  int result = k{name}Map[serialized_value];
  TORCH_CHECK(
      result != kInvalid,
      "Invalid serialized {name} value: ",
      serialized_value);
  return static_cast<c10::{c10_type}>(result);
}}
"""

    scalar_type_converter = generate_converter(
        "ScalarType",
        "ScalarType",
        schema.SCALAR_TYPE_TO_C10,
        max(schema.SCALAR_TYPE_TO_C10.keys()),
    )
    layout_converter = generate_converter(
        "Layout",
        "Layout",
        schema.LAYOUT_TO_C10,
        max(schema.LAYOUT_TO_C10.keys()),
    )
    memory_format_converter = generate_converter(
        "MemoryFormat",
        "MemoryFormat",
        schema.MEMORY_FORMAT_TO_C10,
        max(schema.MEMORY_FORMAT_TO_C10.keys()),
    )

    return f"""
#pragma once

#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

// Converter functions from serialized enum values (torch._export.serde.schema)
// to c10 enums. The serialized format has different enum values than c10.

namespace torch::aot_inductor {{
{scalar_type_converter}
{layout_converter}
{memory_format_converter}
}} // namespace torch::aot_inductor
"""


@dataclasses.dataclass
class _Commit:
    result: dict[str, Any]
    checksum_next: str
    yaml_path: str
    additions: dict[str, Any]
    subtractions: dict[str, Any]
    base: dict[str, Any]
    checksum_head: str | None
    cpp_header: str
    cpp_header_path: str
    enum_converter_header: str
    enum_converter_header_path: str
    thrift_checksum_head: str | None
    thrift_checksum_real: str | None
    thrift_checksum_next: str
    thrift_schema: str
    thrift_schema_path: str


def update_schema():
    import importlib.resources

    # pyrefly: ignore [bad-argument-type]
    if importlib.resources.is_resource(__package__, "schema.yaml"):
        # pyrefly: ignore [bad-argument-type]
        content = importlib.resources.read_text(__package__, "schema.yaml")
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", content)
        _check(match is not None, "checksum not found in schema.yaml")
        if match is None:
            raise AssertionError("checksum not found in schema.yaml")
        checksum_head = match.group(1)

        thrift_content = importlib.resources.read_text(
            # pyrefly: ignore [bad-argument-type]
            __package__,
            "export_schema.thrift",
        )
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", thrift_content)
        _check(match is not None, "checksum not found in export_schema.thrift")
        if match is None:
            raise AssertionError("checksum not found in export_schema.thrift")
        thrift_checksum_head = match.group(1)
        thrift_content = thrift_content.splitlines()
        if not thrift_content[0].startswith("// @" + "generated"):
            raise AssertionError(
                f"expected first line to start with '// @generated', got {thrift_content[0]!r}"
            )
        if not thrift_content[1].startswith("// checksum<<"):
            raise AssertionError(
                f"expected second line to start with '// checksum<<', got {thrift_content[1]!r}"
            )
        thrift_checksum_real = _hash_content("\n".join(thrift_content[2:]))

        from yaml import load, Loader

        dst = load(content, Loader=Loader)
        if not isinstance(dst, dict):
            raise AssertionError(f"expected dict from yaml, got {type(dst)}")
    else:
        checksum_head = None
        thrift_checksum_head = None
        thrift_checksum_real = None
        dst = {"SCHEMA_VERSION": None, "TREESPEC_VERSION": None}

    src, cpp_header, thrift_schema = _staged_schema()
    enum_converter_header = _generate_enum_converters()
    additions, subtractions = _diff_schema(dst, src)
    # pyrefly: ignore [missing-attribute]
    yaml_path = __package__.replace(".", "/") + "/schema.yaml"
    # pyrefly: ignore [missing-attribute]
    thrift_schema_path = __package__.replace(".", "/") + "/export_schema.thrift"
    torch_prefix = "torch/"
    if not yaml_path.startswith(torch_prefix):
        raise AssertionError(
            f"yaml_path must start with {torch_prefix}, got {yaml_path}"
        )
    if not thrift_schema_path.startswith(torch_prefix):
        raise AssertionError(
            f"thrift_schema_path must start with {torch_prefix}, got {thrift_schema_path}"
        )

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
        enum_converter_header=enum_converter_header,
        enum_converter_header_path=torch_prefix
        + "csrc/inductor/aoti_torch/generated_enum_converters.h",
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
                        f"Field {k}.{f} is added to schema.py without a default value as an incompatible change "
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
