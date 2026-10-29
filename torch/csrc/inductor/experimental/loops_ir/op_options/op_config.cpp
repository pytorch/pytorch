#include "op_config.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace torch::inductor::loops_ir::op_options {

namespace {

template <typename Config>
struct IntField {
  const char* name;
  int64_t Config::*member;
  std::vector<int64_t> legal;
  bool tunable;
  const char* meaning;
};

const std::vector<int64_t> kGemmBlocks{16, 32, 64, 128, 256};

const std::vector<IntField<MatmulConfig>>& matmulFields() {
  static const std::vector<IntField<MatmulConfig>> fields{
      {"block_m", &MatmulConfig::block_m, kGemmBlocks, true, "output tile rows"},
      {"block_n", &MatmulConfig::block_n, kGemmBlocks, true, "output tile columns"},
      {"block_k", &MatmulConfig::block_k, kGemmBlocks, true, "K step per dot"},
      {"group_m", &MatmulConfig::group_m, {1, 8}, false, "row-group swizzle of program ids (manifest: 8)"},
      {"num_warps", &MatmulConfig::num_warps, {2, 4, 8}, true, "warps per program"},
      {"num_stages", &MatmulConfig::num_stages, {1, 2, 3, 4, 5}, true, "software pipeline stages"},
      {"split_k", &MatmulConfig::split_k, {1}, false, "K split factor (only 1 implemented)"},
  };
  return fields;
}

const std::vector<IntField<ReductionConfig>>& reductionFields() {
  static const std::vector<IntField<ReductionConfig>> fields{
      {"block_x", &ReductionConfig::block_x, {1, 2, 4, 8, 16, 32, 64, 128}, true, "output rows per program"},
      {"block_r",
       &ReductionConfig::block_r,
       {16, 32, 64, 128, 256, 512, 1024, 2048, 4096},
       true,
       "reduction lanes per step (>= R: single pass)"},
      {"num_warps", &ReductionConfig::num_warps, {1, 2, 4, 8, 16}, true, "warps per program"},
      {"num_stages", &ReductionConfig::num_stages, {1}, false, "pipeline stages (fixed)"},
  };
  return fields;
}

template <typename Config>
std::vector<ParamSpec> intParams(const std::vector<IntField<Config>>& fields) {
  const Config defaults{};
  std::vector<ParamSpec> out;
  for (const auto& f : fields) {
    ParamSpec p;
    p.name = f.name;
    p.kind = ParamKind::Int;
    p.legal_ints = f.legal;
    p.default_int = defaults.*(f.member);
    p.tunable = f.tunable;
    p.meaning = f.meaning;
    out.push_back(std::move(p));
  }
  return out;
}

ParamSpec enumParam(std::string name,
                    std::vector<std::string> legal,
                    std::string def,
                    bool tunable,
                    std::string meaning) {
  ParamSpec p;
  p.name = std::move(name);
  p.kind = ParamKind::Enum;
  p.legal_enums = std::move(legal);
  p.default_enum = std::move(def);
  p.tunable = tunable;
  p.meaning = std::move(meaning);
  return p;
}

ConfigSchema buildMatmulSchema() {
  ConfigSchema s;
  s.kind = ConfigKind::Matmul;
  s.params = intParams(matmulFields());
  s.params.push_back(enumParam("matrix_precision",
                               {"ieee", "tf32"},
                               "ieee",
                               false,
                               "tl.dot input_precision; context-derived: tf32 iff stock settings allow"));
  s.params.push_back(enumParam("acc_dtype", {"f32"}, "f32", false, "accumulator dtype"));
  s.constraints = {
      "num_stages*(block_m*block_k + block_k*block_n)*4 <= context.max_shared_memory_bytes",
      "block_m*block_n <= 32768 (accumulator register budget)",
      "matrix_precision == tf32 requires stockAllowsTf32(signature, context)",
  };
  return s;
}

ConfigSchema buildReductionSchema() {
  ConfigSchema s;
  s.kind = ConfigKind::Reduction;
  s.params = intParams(reductionFields());
  s.params.push_back(enumParam("acc_dtype", {"f32"}, "f32", false, "accumulator dtype"));
  s.constraints = {
      "block_x*block_r <= 8192 (register tile budget)",
      "single reduction dim; block_r >= R selects the single-pass form",
  };
  return s;
}

ConfigSchema buildNativeSchema() {
  ConfigSchema s;
  s.kind = ConfigKind::Native;
  s.params.push_back(
      enumParam("vendor", {"aten_eager_default"}, "aten_eager_default", false, "eager ATen vendor dispatch"));
  return s;
}

template <typename Config>
void checkIntFields(const Config& c, const std::vector<IntField<Config>>& fields, const char* type) {
  for (const auto& f : fields) {
    const int64_t v = c.*(f.member);
    if (std::find(f.legal.begin(), f.legal.end(), v) == f.legal.end()) {
      reject(RejectReason::Config, std::string(type) + "." + f.name + "=" + std::to_string(v) + " outside schema domain");
    }
  }
}

void checkEnum(const ConfigSchema& schema, const char* param, const char* value, const char* type) {
  const auto& legal = schema.param(param).legal_enums;
  if (std::find(legal.begin(), legal.end(), value) == legal.end()) {
    reject(RejectReason::Config, std::string(type) + "." + param + "=" + value + " outside schema domain");
  }
}

template <typename Config>
void appendIntFields(std::ostringstream& os, const Config& c, const std::vector<IntField<Config>>& fields) {
  for (const auto& f : fields) {
    os << f.name << "=" << c.*(f.member) << ",";
  }
}

} // namespace

const char* accDtypeName(AccDtype dtype) {
  return dtype == AccDtype::F32 ? "f32" : "?";
}

const char* nativeVendorName(NativeVendor vendor) {
  return vendor == NativeVendor::AtenEagerDefault ? "aten_eager_default" : "?";
}

const char* configKindName(ConfigKind kind) {
  switch (kind) {
    case ConfigKind::Matmul:
      return "matmul";
    case ConfigKind::Reduction:
      return "reduction";
    case ConfigKind::Native:
      return "native";
  }
  return "?";
}

std::string MatmulConfig::canonical() const {
  std::ostringstream os;
  os << "MatmulConfig(";
  appendIntFields(os, *this, matmulFields());
  os << "matrix_precision=" << matmulPrecisionName(matrix_precision) << ",acc_dtype=" << accDtypeName(acc_dtype)
     << ")";
  return os.str();
}

bool MatmulConfig::operator==(const MatmulConfig& o) const {
  return canonical() == o.canonical();
}

std::string ReductionConfig::canonical() const {
  std::ostringstream os;
  os << "ReductionConfig(";
  appendIntFields(os, *this, reductionFields());
  os << "acc_dtype=" << accDtypeName(acc_dtype) << ")";
  return os.str();
}

bool ReductionConfig::operator==(const ReductionConfig& o) const {
  return canonical() == o.canonical();
}

std::string NativeConfig::canonical() const {
  return std::string("NativeConfig(vendor=") + nativeVendorName(vendor) + ")";
}

bool NativeConfig::operator==(const NativeConfig& o) const {
  return vendor == o.vendor;
}

ConfigKind configKind(const OptionConfig& config) {
  if (std::holds_alternative<MatmulConfig>(config)) {
    return ConfigKind::Matmul;
  }
  if (std::holds_alternative<ReductionConfig>(config)) {
    return ConfigKind::Reduction;
  }
  return ConfigKind::Native;
}

std::string canonicalConfig(const OptionConfig& config) {
  return std::visit([](const auto& c) { return c.canonical(); }, config);
}

const ParamSpec& ConfigSchema::param(const std::string& name) const {
  for (const auto& p : params) {
    if (p.name == name) {
      return p;
    }
  }
  reject(RejectReason::Config, "schema has no parameter '" + name + "'");
}

const ConfigSchema& configSchema(ConfigKind kind) {
  static const ConfigSchema matmul = buildMatmulSchema();
  static const ConfigSchema reduction = buildReductionSchema();
  static const ConfigSchema native = buildNativeSchema();
  switch (kind) {
    case ConfigKind::Matmul:
      return matmul;
    case ConfigKind::Reduction:
      return reduction;
    case ConfigKind::Native:
      return native;
  }
  return native;
}

void validateDomain(const OptionConfig& config) {
  if (const auto* m = std::get_if<MatmulConfig>(&config)) {
    checkIntFields(*m, matmulFields(), "MatmulConfig");
    const ConfigSchema& s = configSchema(ConfigKind::Matmul);
    checkEnum(s, "matrix_precision", matmulPrecisionName(m->matrix_precision), "MatmulConfig");
    checkEnum(s, "acc_dtype", accDtypeName(m->acc_dtype), "MatmulConfig");
  } else if (const auto* r = std::get_if<ReductionConfig>(&config)) {
    checkIntFields(*r, reductionFields(), "ReductionConfig");
    checkEnum(configSchema(ConfigKind::Reduction), "acc_dtype", accDtypeName(r->acc_dtype), "ReductionConfig");
  } else {
    const auto& n = std::get<NativeConfig>(config);
    checkEnum(configSchema(ConfigKind::Native), "vendor", nativeVendorName(n.vendor), "NativeConfig");
  }
}

} // namespace torch::inductor::loops_ir::op_options
