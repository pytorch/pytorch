#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <utility>

namespace torch::jit::tensorexpr {

struct TORCH_API SimplifierHashType {
  SimplifierHashType() = default;
  explicit SimplifierHashType(size_t s) : _h(s) {}

  bool operator==(const SimplifierHashType& other) const;
  bool operator!=(const SimplifierHashType& other) const;
  bool operator<(const SimplifierHashType& other) const;
  bool operator==(const size_t other) const;
  bool operator!=(const size_t other) const;

  size_t _h{0};
};

} // namespace torch::jit::tensorexpr

namespace std {
template <>
struct hash<torch::jit::tensorexpr::SimplifierHashType> {
  size_t operator()(
      const torch::jit::tensorexpr::SimplifierHashType& k) const noexcept {
    return k._h;
  }
};

} // namespace std

namespace torch::jit::tensorexpr {

#define CACHE_GUARD()  \
  if (cachedHash(v)) { \
    return;            \
  }

class Term;
class Polynomial;

/* Expression hasher providing comparable values representing sub-exprs.
 * Uses memoization to avoid excessive recursion. */
class TORCH_API HashProvider : public IRVisitor {
 public:
  template <class T>
  SimplifierHashType hash(T e) {
    e->accept(this);
    return hashOf(e);
  }

  bool cachedHash(const ExprPtr& e) {
    return exprToHash_.find(e) != exprToHash_.end();
  }
  bool cachedHash(const StmtPtr& s) {
    return stmtToHash_.find(s) != stmtToHash_.end();
  }

  void clearCache() {
    exprToHash_.clear();
    stmtToHash_.clear();
  }

  void visit(const AddPtr& v) override;
  void visit(const SubPtr& v) override;
  void visit(const MulPtr& v) override;
  void visit(const DivPtr& v) override;
  void visit(const ModPtr& v) override;
  void visit(const RoundOffPtr& v) override;
  void visit(const MaxPtr& v) override;
  void visit(const MinPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;

#define IMM_VISIT(Type, Name)                    \
  void visit(const Name##ImmPtr& v) override {   \
    CACHE_GUARD();                               \
    putHash(v, hash_combine(#Name, v->value())); \
  }
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT)
#undef IMM_VISIT

  void visit(const CastPtr& v) override;
  void visit(const VarPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const BroadcastPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;
  void visit(const AllocatePtr& v) override;
  void visit(const FreePtr& v) override;
  void visit(const CondPtr& v) override;
  void visit(const TermPtr& v) override;
  void visit(const PolynomialPtr& v) override;
  void visit(const MaxTermPtr& v) override;
  void visit(const MinTermPtr& v) override;

  template <typename... Types>
  SimplifierHashType hash_combine(const Types&... args) {
    SimplifierHashType seed;
    _hash_combine(seed, args...);
    return seed;
  }

 private:
  SimplifierHashType hashOf(const ExprPtr& e) {
    auto it = exprToHash_.find(e);
    if (it != exprToHash_.end()) {
      return it->second;
    }

    // As a failsafe fall back to IRPrinter.
    std::stringstream ss;
    IRPrinter printer(ss);
    e->accept(&printer);
    SimplifierHashType hash = SimplifierHashType(te_hash(ss.str()));
    putHash(e, hash);

    return hash;
  }

  SimplifierHashType hashOf(const StmtPtr& s) {
    auto it = stmtToHash_.find(s);
    if (it != stmtToHash_.end()) {
      return it->second;
    }

    // As a failsafe fall back to IRPrinter.
    std::stringstream ss;
    IRPrinter printer(ss);
    s->accept(&printer);
    SimplifierHashType hash = SimplifierHashType(te_hash(ss.str()));
    putHash(s, hash);

    return hash;
  }

  // Hash funcs for various types, numbers are random.
  template <typename T>
  void _hash_combine(SimplifierHashType& seed, const T& val) {
    seed._h ^= te_hash(val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
  }

  void _hash_combine(SimplifierHashType& seed, const char* val) {
    seed._h ^= te_hash(val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
  }

  // at:::Half doesn't have a prime_number_hash, so cast to short.
  void _hash_combine(SimplifierHashType& seed, const at::Half& val) {
    seed._h ^=
        te_hash((uint16_t)val) + 0x1f752c19 + (seed._h << 7) + (seed._h >> 4);
  }

  void _hash_combine(SimplifierHashType& seed, const Dtype& val) {
    seed._h ^= te_hash(val.ToCppString()) + 0x1f752c19 + (seed._h << 7) +
        (seed._h >> 4);
  }

  void _hash_combine(SimplifierHashType& seed, ExprPtr e) {
    _hash_combine(seed, hash(std::move(e)));
  }

  template <typename T, typename... Types>
  void _hash_combine(
      SimplifierHashType& seed,
      const T& val,
      const Types&... args) {
    _hash_combine(seed, val);
    _hash_combine(seed, args...);
  }

  void putHash(const ExprPtr& e, SimplifierHashType h) {
    auto res = exprToHash_.emplace(e, h);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }
  void putHash(const StmtPtr& s, SimplifierHashType h) {
    auto res = stmtToHash_.emplace(s, h);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

  std::unordered_map<ExprPtr, SimplifierHashType> exprToHash_;
  std::unordered_map<StmtPtr, SimplifierHashType> stmtToHash_;
  UniqueNameManager name_manager_;

  size_t te_hash(SimplifierHashType val) {
    return val._h;
  }

  size_t te_hash(int64_t val) {
    // put the thing down.
    size_t h = val ^ 0x647AA4D20C0B;
    // bit flip it.
    size_t h2 = ~h;
    // and reverse byte order.
    size_t h3 = 0;
    for (unsigned int i = 0; i < 64; i += 8) {
      h3 |= ((h2 >> i) & 0xFF) << (64 - i - 8);
    }
    return h3;
  }

  size_t te_hash(int32_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  size_t te_hash(uint32_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  size_t te_hash(uint64_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  size_t te_hash(int16_t val) {
    int64_t v2 = val;
    return te_hash(v2);
  }

  size_t te_hash(std::string val) {
    size_t hash{0};
    int64_t intval{0};
    int64_t s = val.size() - 1;
    while (s >= 0) {
      for (unsigned int i = 0; i < 8; ++i) {
        if (s < 0)
          break;
        int64_t c = val[s];
        intval |= (c << (i * 8));

        s--;
      }
      hash ^= te_hash(intval);
      intval = 0;
    }

    return hash;
  }

  size_t te_hash(double d) {
    int64_t* n = reinterpret_cast<int64_t*>(&d);
    return te_hash(*n);
  }

  size_t te_hash(float d) {
    int32_t* n = reinterpret_cast<int32_t*>(&d);
    return te_hash(*n);
  }

  size_t te_hash(at::Half d) {
    int16_t* n = reinterpret_cast<int16_t*>(&d);
    return te_hash(*n);
  }

  size_t te_hash(at::BFloat16 d) {
    int16_t* n = reinterpret_cast<int16_t*>(&d);
    return te_hash(*n);
  }
};

} // namespace torch::jit::tensorexpr
