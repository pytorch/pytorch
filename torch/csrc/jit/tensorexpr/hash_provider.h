#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {
template <>
struct hash<torch::jit::tensorexpr::SimplifierHashType> {
  size_t operator()(const torch::jit::tensorexpr::SimplifierHashType& k) const {
    return k._h;
  }
};

} // namespace std

namespace torch {
namespace jit {
namespace tensorexpr {

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
  SimplifierHashType hash(const T* e) {
    e->accept(this);
    return hashOf(e);
  }

  bool cachedHash(const KernelScopedObject* e) {
    return exprToHash_.find(e) != exprToHash_.end();
  }

  void clearCache() {
    exprToHash_.clear();
  }

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Mod* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;

// NOLINTNEXTLINE
#define IMM_VISIT(Type, Name)                    \
  void visit(const Name##Imm* v) override {      \
    CACHE_GUARD();                               \
    putHash(v, hash_combine(#Name, v->value())); \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

  void visit(const Cast* v) override;
  void visit(const Var* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const Block* v) override;
  void visit(const For* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Cond* v) override;
  void visit(const Term* v) override;
  void visit(const Polynomial* v) override;
  void visit(const MaxTerm* v) override;
  void visit(const MinTerm* v) override;

  template <typename... Types>
  SimplifierHashType hash_combine(const Types&... args) {
    SimplifierHashType seed;
    _hash_combine(seed, args...);
    return seed;
  }

 private:
  SimplifierHashType hashOf(const Expr* e) {
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

  SimplifierHashType hashOf(const Stmt* s) {
    auto it = exprToHash_.find(s);
    if (it != exprToHash_.end()) {
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

  void _hash_combine(SimplifierHashType& seed, const Expr* e) {
    _hash_combine(seed, hash(e));
  }

  template <typename T, typename... Types>
  void _hash_combine(
      SimplifierHashType& seed,
      const T& val,
      const Types&... args) {
    _hash_combine(seed, val);
    _hash_combine(seed, args...);
  }

  void putHash(const KernelScopedObject* e, SimplifierHashType h) {
    auto res = exprToHash_.emplace(e, h);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

  std::unordered_map<const KernelScopedObject*, SimplifierHashType> exprToHash_;
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
    int s = val.size() - 1;
    while (s >= 0) {
      for (unsigned int i = 0; i < 8; ++i) {
        if (s < 0)
          break;
        int64_t c = val.data()[s];
        intval |= (c << (i * 8));

        s--;
      }
      hash ^= te_hash(intval);
      intval = 0;
    }

    return hash;
  }

  size_t te_hash(double d) {
    // memcpy as type punning. Should be optimized out.
    int64_t n;
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }

  size_t te_hash(float d) {
    // memcpy as type punning. Should be optimized out.
    int32_t n;
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }

  size_t te_hash(at::Half d) {
    // memcpy as type punning. Should be optimized out.
    int16_t n;
    std::memcpy(&n, &d, sizeof d);
    return te_hash(n);
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
