#pragma once
// Typed per-option configurations and their visible schemas. Every tunable
// and fixed parameter is a named struct field with an explicit finite
// domain; there is no keyword bag and no hidden global parameter.

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "op_types.h"

namespace torch::inductor::loops_ir::op_options {

enum class AccDtype { F32 };
enum class NativeVendor { AtenEagerDefault };
enum class ConfigKind { Matmul, Reduction, Native };
const char* configKindName(ConfigKind kind);
const char* accDtypeName(AccDtype dtype);
const char* nativeVendorName(NativeVendor vendor);

struct MatmulConfig {
  int64_t block_m = 64;
  int64_t block_n = 64;
  int64_t block_k = 32;
  int64_t group_m = 8;
  int64_t num_warps = 4;
  int64_t num_stages = 3;
  int64_t split_k = 1;
  MatmulPrecision matrix_precision = MatmulPrecision::Ieee;
  AccDtype acc_dtype = AccDtype::F32;
  std::string canonical() const;
  bool operator==(const MatmulConfig& o) const;
};

// block_r >= R selects the single-pass (persistent) form; smaller block_r
// selects the looped form. Both are explicit consequences of block_r.
struct ReductionConfig {
  int64_t block_x = 1;
  int64_t block_r = 1024;
  int64_t num_warps = 4;
  int64_t num_stages = 1;
  AccDtype acc_dtype = AccDtype::F32;
  std::string canonical() const;
  bool operator==(const ReductionConfig& o) const;
};

// Native options delegate to the eager vendor dispatch; nothing is tunable.
struct NativeConfig {
  NativeVendor vendor = NativeVendor::AtenEagerDefault;
  std::string canonical() const;
  bool operator==(const NativeConfig& o) const;
};

using OptionConfig = std::variant<MatmulConfig, ReductionConfig, NativeConfig>;
ConfigKind configKind(const OptionConfig& config);
std::string canonicalConfig(const OptionConfig& config);

enum class ParamKind { Int, Enum };

// legal_* is the full accepted domain; `tunable` means the option's
// candidate manifest varies this parameter (otherwise every generated
// candidate uses the default or a context-derived value).
struct ParamSpec {
  std::string name;
  ParamKind kind = ParamKind::Int;
  std::vector<int64_t> legal_ints;
  std::vector<std::string> legal_enums;
  int64_t default_int = 0;
  std::string default_enum;
  bool tunable = true;
  std::string meaning;
};

struct ConfigSchema {
  ConfigKind kind = ConfigKind::Matmul;
  std::vector<ParamSpec> params;
  // Cross-field/context constraints enforced by the owning option.
  std::vector<std::string> constraints;
  const ParamSpec& param(const std::string& name) const;
};

const ConfigSchema& configSchema(ConfigKind kind);

// Every int and enum field must lie in its schema domain (invalid enum
// values included). Throws OptionRejected(Config).
void validateDomain(const OptionConfig& config);

} // namespace torch::inductor::loops_ir::op_options
