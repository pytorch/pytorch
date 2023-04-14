#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>
#include <torch/csrc/lazy/core/cache.h>

#include <utility>

// SHAPE CACHING CODE
namespace torch {
namespace jit {
namespace {
using CanonicalArg = c10::variant<CanonicalizedSymbolicShape, IValue>;
using CanonicalArgVec = std::vector<CanonicalArg>;
using CanonicalRet = std::vector<CanonicalizedSymbolicShape>;
using ShapeCacheKey = std::tuple<c10::OperatorName, CanonicalArgVec>;

CanonicalArgVec cannonicalizeVec(
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args;
  canonical_args.reserve(arg_vec.size());
  for (auto& arg : arg_vec) {
    if (const IValue* iv = c10::get_if<IValue>(&arg)) {
      if (deep_copy) {
        canonical_args.emplace_back(iv->deepcopy());
      } else {
        canonical_args.emplace_back(*iv);
      }
    } else {
      auto& ss = c10::get<at::SymbolicShape>(arg);
      canonical_args.emplace_back(CanonicalizedSymbolicShape(ss, ss_map));
    }
  }
  return canonical_args;
}

std::vector<CanonicalizedSymbolicShape> cannonicalizeVec(
    const std::vector<at::SymbolicShape>& ret_vec,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  std::vector<CanonicalizedSymbolicShape> canonical_rets;
  canonical_rets.reserve(ret_vec.size());
  for (auto& ss : ret_vec) {
    canonical_rets.emplace_back(ss, ss_map);
  }
  return canonical_rets;
}

struct ArgumentsHasher {
  size_t operator()(const ShapeCacheKey& cacheKey) const {
    // TODO: ignore arguments that are not used in shape function (not needed
    // initially)
    auto& op_name = std::get<0>(cacheKey);
    auto& arg_vec = std::get<1>(cacheKey);

    size_t hash_val = c10::hash<c10::OperatorName>()(op_name);

    hash_val = at::hash_combine(std::hash<size_t>{}(arg_vec.size()), hash_val);
    for (const CanonicalArg& arg : arg_vec) {
      size_t cur_arg = 0;
      if (const IValue* ival = c10::get_if<IValue>(&arg)) {
        // IValue doesn't hash List (as Python doesn't), so we will do a custom
        // list hash
        if (ival->isList()) {
          TORCH_INTERNAL_ASSERT(ival->isIntList(), "Unexpected Args in List");
          cur_arg = ival->toListRef().size();
          for (const IValue& elem_ival : ival->toListRef()) {
            cur_arg = at::hash_combine(cur_arg, IValue::hash(elem_ival));
          }
        } else {
          cur_arg = IValue::hash(ival);
        }
      } else {
        cur_arg = c10::get<CanonicalizedSymbolicShape>(arg).hash();
      }
      hash_val = at::hash_combine(hash_val, cur_arg);
    }
    return hash_val;
  }
};

using ShapeCache = lazy::Cache<
    ShapeCacheKey,
    std::vector<CanonicalizedSymbolicShape>,
    ArgumentsHasher>;

constexpr size_t kShapeCacheSize = 1024;
ShapeCache shapeCache(kShapeCacheSize);

ShapeCacheKey get_cache_key(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args = cannonicalizeVec(arg_vec, ss_map, deep_copy);
  return std::make_tuple(schema->operator_name(), canonical_args);
}

} // namespace

TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>>
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key = get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ true);
  auto can_ret_vec = std::make_shared<std::vector<CanonicalizedSymbolicShape>>(
      cannonicalizeVec(ret_vec, ss_map));
  shapeCache.Add(std::move(cache_key), std::move(can_ret_vec));
}

TORCH_API c10::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>> for both
  // ss_map and inverse_ss_map
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key =
      get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ false);
  auto cached_ret_vec = shapeCache.Get(cache_key);
  if (cached_ret_vec == nullptr) {
    return c10::nullopt;
  }
  // Decanonicalize the return values
  auto inverse_ss_map = std::unordered_map<int64_t, int64_t>();
  for (auto& ss_val : ss_map) {
    inverse_ss_map[ss_val.second] = ss_val.first;
  }
  std::vector<at::SymbolicShape> ret_vec;
  for (auto& css : *cached_ret_vec) {
    ret_vec.emplace_back(css.toSymbolicShape(inverse_ss_map));
  }
  return ret_vec;
}

// Function only to access the cache, used for testing
TORCH_API void clear_shape_cache() {
  shapeCache.Clear();
}

TORCH_API size_t get_shape_cache_size() {
  return shapeCache.Numel();
}

void CanonicalizedSymbolicShape::init(
    const c10::SymbolicShape& orig_shape,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  auto sizes = orig_shape.sizes();
  if (!sizes) {
    values_ = c10::nullopt;
    return;
  }
  values_ = std::vector<int64_t>();
  int64_t cur_symbolic_index = -static_cast<int64_t>(ss_map.size()) - 1;
  for (auto& cur_shape : *sizes) {
    if (cur_shape.is_static()) {
      values_->push_back(cur_shape.static_size());
    } else {
      // Check for aliasing
      auto it = ss_map.find(cur_shape.value());

      if (it == ss_map.end()) {
        values_->push_back(cur_symbolic_index);
        ss_map.insert({cur_shape.value(), cur_symbolic_index});
        cur_symbolic_index--;
      } else {
        values_->push_back(it->second);
      }
    }
  }
}

c10::SymbolicShape CanonicalizedSymbolicShape::toSymbolicShape(
    std::unordered_map<int64_t, int64_t>& inverse_ss_map) const {
  if (!values_.has_value()) {
    return c10::SymbolicShape();
  }
  std::vector<at::ShapeSymbol> sizes;
  for (long long cur_val : *values_) {
    if (cur_val >= 0) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(cur_val));
      continue;
    }
    auto res = inverse_ss_map.find(cur_val);
    if (res != inverse_ss_map.end()) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(res->second));
    } else {
      auto new_symbol = at::ShapeSymbol::newSymbol();
      inverse_ss_map.insert({cur_val, new_symbol.value()});
      sizes.push_back(new_symbol);
    }
  }
  return c10::SymbolicShape(std::move(sizes));
}

size_t CanonicalizedSymbolicShape::hash() const {
  if (!values_.has_value()) {
    return 0x8cc80c80; // random value to prevent hash collisions
  }
  return c10::hash<std::vector<int64_t>>()(values_.value());
}

bool operator==(
    const CanonicalizedSymbolicShape& a,
    const CanonicalizedSymbolicShape& b) {
  return a.values_ == b.values_;
};
} // namespace jit
} // namespace torch
