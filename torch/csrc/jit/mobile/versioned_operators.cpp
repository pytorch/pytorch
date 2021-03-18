#include <c10/util/Optional.h>
#include <torch/csrc/jit/mobile/versioned_operators.h>

#include <unordered_map>

// This file is to handle operator versioning in mobile.
// How does it work?
// At top level, with a tuple of (operator_name, op_version_in_model,
// bytecode_version_in_model), check the compatibility for that operator. If
// it's not compatible, throw an error with reason. If it's compatible, do
// compatibility adaptation if necessary and return the corresponding operator
// functor that is compatible to this runtime. The logic is implemented in
// function operator_resolver().
//
// Under the hood, a static table,
// op_version_table, is saved for current runtime. The keys are the operator
// names and values are a container with version numbers and their metadata. The
// incoming operator with version is searched in the table to find the match.
// Note that the size of op_version_table is small because it only contains
// operators with compatibility updates. For most of the operators, they
// fall through to the existing implementation with default version 0.

namespace torch {
namespace jit {
namespace mobile {
namespace {

// The version info is limited to combinations of two situations:
//   * Route to another op name
//   * Push additional default values
// It’s fine to update the content of version info, to handle more situations in
// future. Since it’s implementation details, it should not affect compatibility.
struct VersionInfo {
  VersionInfo(
      const c10::OperatorName& name,
      c10::optional<IValue> default_val)
      : name(name), default_val(default_val) {}
  c10::OperatorName name;
  c10::optional<IValue> default_val;
};

// Auxiliary function for operator dispatch from name
OperatorFunctor findOperatorFromName(const c10::OperatorName& opname) {
  auto jit_op = findOperatorFor(opname);
  OperatorFunctor fn;
  if (jit_op) {
    fn = [jit_op](Stack& stack) { jit_op->getOperation()(&stack); };
  } else {
    auto op = c10::Dispatcher::singleton().findSchema(opname);
    if (op.has_value()) {
      fn = [op](Stack& stack) { op->callBoxed(&stack); };
    } else {
      return fn;
    }
  }
  return nullptr;
}

// VersionInfoContainer holds a map of {verion_num: VersionInfo} and an explicit
// cur_version. cur_version is necessary for the situation where an update
// invalidate all versions in the container. In such case there's an
// empty version_vec and the only supported version is cur_version.
// an example of VersionInfoContainer
//   version_map =
//     op_version  | op_info
//     0           | {"op_A_old", None}
//     1           | {"op_A", default_val}
//   cur_version = 2
struct VersionInfoContainer {
  VersionInfoContainer(std::unordered_map<int64_t, VersionInfo> version_map, int64_t cur_version)
      : version_map(version_map), cur_version(cur_version) {}

  // Resolve the operator from the op_version. name is used for messaging.
  OperatorFunctor resolve_operator(const c10::OperatorName& name, int64_t op_version) const {
    if (op_version == cur_version) {
      // It's the current version, no adaptation is needed.
      return findOperatorFromName(name);
    }
    auto it = version_map.find(op_version);
    if (it != version_map.end()) {
      // The version is in the supported map, do corresponding adaptation and return
      const auto& opinfo = it->second;
      auto fn = findOperatorFromName(opinfo.name);
      if (opinfo.default_val) {
        fn = [fn, opinfo](Stack& stack) {
          stack.push_back(opinfo.default_val.value());
          fn(stack);
        };
      }
      return fn;
    }
    // The version is not supported, throw error with the original name
    TORCH_CHECK(
        false,
        "The version number of ",
        c10::toString(name),
        ", ",
        op_version,
        " is not compatible in this runtime. ");
    return nullptr;
  }

  std::unordered_map<int64_t, VersionInfo> version_map;
  int64_t cur_version;
};

// The operator version table that contains the operators with versions that
// this runtime can "up-version". "up-version" here means that if there's an
// operator with older version in the model, some treatment can be done to
// make this model run with current (newer) version of that operator. Currently,
// this "up-version" is limited to combinations of two situations:
//   * Route to another op name
//   * Push additional default values
// This limitation is guarded by the definition of the VersionInfo struct.
static const std::unordered_map<std::string, VersionInfoContainer>
    op_version_table(
        {
          {"aten::_convolution",
            VersionInfoContainer(
              {{0, VersionInfo(c10::OperatorName("aten::_convolution", ""), true)}}, 1)
          }
        });
} // namespace

OperatorFunctor operator_resolver(
    const c10::OperatorName& opname,
    int64_t op_version,
    int64_t model_version) {
  if (model_version > 0x3LL) {
    auto it = op_version_table.find(toString(opname));
    if (it == op_version_table.end()) {
      // Not in the version table, by default it fall through the original
      // opname with no compatibility treatment
      return findOperatorFromName(opname);
    } else {
      return it->second.resolve_operator(opname, op_version);
    }
  } else if (model_version == 0x3LL) {
    auto fn = findOperatorFromName(opname);
    if (opname == c10::OperatorName("aten::_convolution", "")) {
      // Since byte-code versions 0x4L, convolution has an additional
      // default-value argument (allow_tf32=True, see
      // https://github.com/pytorch/pytorch/pull/40737). This wrapper handles
      // backward compatibility with models of byte-code version <= 0x3L, where
      // this bool argument does not yet exist.
      fn = [fn](Stack& stack) {
        stack.push_back(true);
        fn(stack);
      };
    }
    return fn;
  }
  return nullptr;
}

std::unordered_map<std::string, std::unordered_set<int64_t>>
get_op_version_table() {
  std::unordered_map<std::string, std::unordered_set<int64_t>> table;
  for (const auto& item : op_version_table) {
    table[item.first].emplace(item.second.cur_version);
    for (const auto& version : item.second.version_map) {
      table[item.first].emplace(version.first);
    }
  }
  return table;
}
} // namespace mobile
} // namespace jit
} // namespace torch
