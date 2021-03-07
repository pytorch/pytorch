#include <c10/util/Optional.h>
#include <torch/csrc/jit/mobile/versioned_operators.h>

#include <unordered_map>

// This file is to handle operator versioning in mobile.
// How does it work?
// At top level, with a tuple of (operator_name, op_version_in_model,
// bytecode_version_in_model), check the compatibility for that operator. If
// it's not compatible, throw an error with reason. If it's compatible, do
// compatibility treatments if necessary and return the corresponding operator
// functor that is compatible to this runtime. The logic is implemented in
// function operator_resolver().
//
// Under the hood, a static table,
// op_version_table, is saved for current runtime. The keys are the operator
// names and values are a container with version numbers and their metadata. The
// incoming operator with version is searched in the table to find the match.
// Note that the size of op_version_table is small because it only contains
// operators that have more than one versions. For most of the operators, they
// fall through to the existing implementation with default version 0.

namespace torch {
namespace jit {
namespace mobile {
namespace {

// The version info is limited to combinations of two situations:
//   * Route to another op name
//   * Push additional default values
// It’s fine to update the content of version info, to handle more situations in
// fugure. Since it’s implementation details, it should not affect
// compatibility.
struct VersionInfo {
  VersionInfo(
      int64_t version,
      const c10::OperatorName& name,
      c10::optional<IValue> default_val)
      : version(version), name(name), default_val(default_val) {}
  int64_t version;
  c10::OperatorName name;
  c10::optional<IValue> default_val;
};

struct VersionInfoContainer {
  VersionInfoContainer(const std::vector<VersionInfo>& version_vec)
      : version_vec(version_vec) {}
  int64_t min_version() const {
    return version_vec.begin()->version;
  }

  int64_t max_version() const {
    return version_vec.rbegin()->version + 1;
  }

  std::vector<VersionInfo> version_vec;
};

VersionInfo getVersioninfo(
    const VersionInfoContainer& container,
    const c10::OperatorName& name,
    int64_t op_version) {
  // Backward compatibility breakage
  TORCH_CHECK(
      op_version >= container.min_version(),
      "The version number of ",
      c10::toString(name),
      ", ",
      op_version,
      ", is smaller than the minimum version number, ",
      container.min_version(),
      ", supported in this runtime. The model file is too old. ",
      "Update the model with most recent code.");

  // Forward compatibility breakage
  TORCH_CHECK(
      op_version <= container.max_version(),
      "The version number of ",
      c10::toString(name),
      ", ",
      op_version,
      ", is larger than the maximum version number, ",
      container.max_version(),
      ", supported in this runtime. Update the runtime to be compatible",
      " to this model.");

  for (const auto& info : container.version_vec) {
    if (info.version == op_version) {
      return info;
    }
  }

  if (op_version == container.max_version() + 1) {
    // Current version
    return VersionInfo(op_version, name, c10::nullopt);
  }

  // The situation where op_version is in [min_version, max_version + 1], but
  // not in the container, which should not happen.
  TORCH_CHECK(
      true,
      "The version number of ",
      c10::toString(name),
      ", ",
      op_version,
      " is not compatible in this runtime. ");

  return VersionInfo(
      0, c10::OperatorName("aten::_convolution", ""), c10::nullopt);
}

static const std::unordered_map<std::string, VersionInfoContainer>
    op_version_table(
        {{"aten::_convolution",
          VersionInfoContainer(
              {{0, c10::OperatorName("aten::_convolution", ""), true}})}});
} // namespace

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
      auto opinfo = getVersioninfo(it->second, opname, op_version);
      auto fn = findOperatorFromName(opinfo.name);
      if (opinfo.default_val) {
        fn = [fn, opinfo](Stack& stack) {
          stack.push_back(opinfo.default_val.value());
          fn(stack);
        };
      }
      return fn;
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

} // namespace mobile
} // namespace jit
} // namespace torch
