#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

namespace {
// Make sure we can inline something, before we attempt to.
void check_inlineable(const IRInputOutput* const irio) {
  for (auto inp : irio->inputs())
    TORCH_CHECK(
        inp->isScalar(),
        "Printing inline computations involving values other than scalars is not currently supported.");
  TORCH_CHECK(
      irio->nOutputs() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      irio->output(0)->isScalar(),
      "Printing inline computations involving values other than scalars is not currently supported.");
}
} // namespace

void IRPrinter::printHeader(Fusion* fusion, const std::string& kernel_name_) {
  // Helper funtions
  os << "class Philox {\n"
     << "public:\n"
     << "  __device__ inline Philox(unsigned long long seed,\n"
     << "                           unsigned long long subsequence,\n"
     << "                           unsigned long long offset) {\n"
     << "    key.x = (unsigned int)seed;\n"
     << "    key.y = (unsigned int)(seed >> 32);\n"
     << "    counter = make_uint4(0, 0, 0, 0);\n"
     << "    counter.z = (unsigned int)(subsequence);\n"
     << "    counter.w = (unsigned int)(subsequence >> 32);\n"
     << "    STATE = 0;\n"
     << "    incr_n(offset / 4);\n"
     << "  }\n"
     << "  __device__ inline unsigned long operator()() {\n"
     << "    if(STATE == 0) {\n"
     << "      uint4 counter_ = counter;\n"
     << "      uint2 key_ = key;\n"
     << "      for(int i = 0; i < 9; i++) {\n"
     << "        counter_ = single_round(counter_, key_);\n"
     << "        key_.x += (kPhilox10A); key_.y += (kPhilox10B);\n"
     << "      }\n"
     << "      output = single_round(counter_, key_);\n"
     << "      incr();\n"
     << "    }\n"
     << "    unsigned long ret;\n"
     << "    switch(STATE) {\n"
     << "      case 0: ret = output.x; break;\n"
     << "      case 1: ret = output.y; break;\n"
     << "      case 2: ret = output.z; break;\n"
     << "      case 3: ret = output.w; break;\n"
     << "    }\n"
     << "    STATE = (STATE + 1) % 4;\n"
     << "    return ret;\n"
     << "  }\n"
     << "private:\n"
     << "  uint4 counter;\n"
     << "  uint4 output;\n"
     << "  uint2 key;\n"
     << "  unsigned int STATE;\n"
     << "  __device__ inline void incr_n(unsigned long long n) {\n"
     << "    unsigned int nlo = (unsigned int)(n);\n"
     << "    unsigned int nhi = (unsigned int)(n >> 32);\n"
     << "    counter.x += nlo;\n"
     << "    if (counter.x < nlo)\n"
     << "      nhi++;\n"
     << "    counter.y += nhi;\n"
     << "    if (nhi <= counter.y)\n"
     << "      return;\n"
     << "    if (++counter.z)\n"
     << "      return;\n"
     << "    ++counter.w;\n"
     << "  }\n"
     << "  __device__ inline void incr() {\n"
     << "    if (++counter.x)\n"
     << "      return;\n"
     << "    if (++counter.y)\n"
     << "      return;\n"
     << "    if (++counter.z)\n"
     << "      return;\n"
     << "    ++counter.w;\n"
     << "  }\n"
     << "  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,\n"
     << "                                    unsigned int *result_high) {\n"
     << "    *result_high = __umulhi(a, b);\n"
     << "    return a*b;\n"
     << "  }\n"
     << "  __device__ inline uint4 single_round(uint4 ctr, uint2 key) {\n"
     << "    unsigned int hi0;\n"
     << "    unsigned int hi1;\n"
     << "    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);\n"
     << "    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);\n"
     << "    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};\n"
     << "    return ret;\n"
     << "  }\n"
     << "  static const unsigned long kPhilox10A = 0x9E3779B9;\n"
     << "  static const unsigned long kPhilox10B = 0xBB67AE85;\n"
     << "  static const unsigned long kPhiloxSA = 0xD2511F53;\n"
     << "  static const unsigned long kPhiloxSB = 0xCD9E8D57;\n"
     << "};\n"
     << "// Inverse of 2^32.\n"
     << "#define M_RAN_INVM32 2.3283064e-10f\n"
     << "__device__  __inline__ float uniform(unsigned int x) {\n"
     << "  return x * M_RAN_INVM32;\n"
     << "}\n"
     << "__device__ int ceilDiv(const int a, const int b) {\n"
     << "  return (a + b - 1) / b;\n"
     << "}\n\n"
     << "__device__ float clamp(const float x, const float minv, const float maxv) {\n"
     << "  return x < minv ? minv : (x > maxv ? maxv : x);\n"
     << "}\n\n"
     << "__device__ float frac(const float x) {\n"
     << "  return x - truncf(x);\n"
     << "}\n\n"
     << "__device__ float gelu(const float x) {\n"
     << "  return 0.5f * x * (1.f + tanhf(sqrtf(3.14159274101 / 2.f) * (x + 0.044715 * powf(x,3.f))));\n"
     << "}\n\n"
     << "__device__ float reciprocal(const float x) {\n"
     << "  return 1.f / x;\n"
     << "}\n\n"
     << "__device__ float relu(const float x) {\n"
     << "  return x <= 0.f ? 0.f : x;\n"
     << "}\n\n"
     << "__device__ float remainder(const float a, const float b) {\n"
     << "  return a - b * floorf(a / b);\n"
     << "}\n\n"
     << "__device__ float sigmoid(const float x) {\n"
     << "  return 1.f / (1.f + expf(-x));\n"
     << "}\n\n"
     << "__device__ float threshold(const float x, const float t, const float v) {\n"
     << "  return x <= t ? v : x;\n"
     << "}\n\n"
     << "__device__ float where(const int c, const float a, const float b) {\n"
     << "  return c ? a : b;\n"
     << "}\n\n"
     << "__device__ float randLike(Philox rnd) {\n"
     << "  return uniform(rnd());\n"
     << "}\n\n";

  os << "__global__ void " << kernel_name_ << "(";

  std::deque<Val*> vals;
  for (decltype(fusion->nInputs()) i{0}; i < fusion->nInputs(); i++)
    vals.push_back(fusion->input(i));
  for (decltype(fusion->nOutputs()) i{0}; i < fusion->nOutputs(); i++)
    vals.push_back(fusion->output(i));

  for (Val* val : vals) {
    switch (val->getValType().value()) {
      case (ValType::TensorView):
        os << "Tensor<" << val->getDataType().value() << ", "
           << static_cast<TensorView*>(val)->getRootDomain()->nDims() << "> T"
           << val->name();
        break;
      case (ValType::Scalar):
        os << val->getDataType().value() << " " << val;
        break;
      default:
        TORCH_CHECK(
            false,
            "printHeader() found an input to the fusion of unexpected data type.");
    }

    if (val != vals.back())
      os << ", ";
  }

  if (fusion->random())
    os << ", unsigned long long seed, unsigned long long offset";
  os << "){\n";
  indent_size++;
  if (fusion->random()) {
    indent(); os << "int idx = blockIdx.x*blockDim.x + threadIdx.x;\n";
    indent(); os << "Philox rnd(seed, idx, offset);\n";
  }
}

void IRPrinter::handle(Fusion* fusion) {
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    handle(expr);
  }
}

void IRPrinter::handle(const TensorDomain* const td) {
  os << "[ ";
  for (std::vector<const IterDomain*>::size_type i = 0; i < td->nDims(); i++) {
    handle(td->axis(i));
    if (i != td->nDims() - 1)
      os << ", ";
  }
  os << " ]";
}

void IRPrinter::handle(const TensorView* const tv) {
  os << "T" << tv->name();
  handle(tv->domain());

  if (tv->getComputeAtView() != nullptr) {
    os << " compute_at( ";
    os << "T" << tv->getComputeAtView()->name();
    os << ", " << tv->getComputeAtAxis() << " )";
  }
}

void IRPrinter::handle(const IterDomain* const id) {
  if (id->isReduction())
    os << "r";
  else
    os << "i";
  switch (id->parallel_method()) {
    case (ParallelType::Vectorize):
      os << "V";
      break;
    case (ParallelType::Unroll):
      os << "U";
      break;
    case (ParallelType::Serial):
      os << "S";
      break;
    default:
      os << id->parallel_method();
  }

  os << "{";
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os << " : ";
  }
  print_inline(id->extent());
  os << "}";
}

void IRPrinter::handle(const TensorIndex* const ti) {
  os << "T" << ti->view()->name() << "[ ";

  bool first = true;
  for (auto* ind : ti->indices()) {
    if (!first)
      os << " + ";
    print_inline(ind);
    first = false;
  }
  os << " ]";
}

void IRPrinter::handle(const TensorContiguity* const t) {
  os << "format_tag: " << t->getContiguityTag();
}

void IRPrinter::handle(const Float* const f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(f));
    os << " )";
    return;
  }

  if (f->isSymbolic()) {
    os << "f" << f->name();
  } else {
    os << "float(" << *(f->value()) << ")";
  }
}

void IRPrinter::handle(const Int* const i) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(i) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(i));
    os << " )";
    return;
  }

  if (i->isSymbolic()) {
    os << "i" << i->name();
  } else {
    os << *(i->value());
  }
}

void IRPrinter::handle(const NamedScalar* const i) {
  os << i->name();
}

namespace {

bool isTV(const Val* const val) {
  return (
      val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex);
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* const expr) {
  if (expr->nOutputs() == 1 && isTV(expr->output(0)))
    return true;
  return false;
}
} // namespace

void IRPrinter::handle(const UnaryOp* const uop) {
  bool istvop = isTVOp(uop);
  if (!print_inline_) {
    indent();
    os << uop->out();
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }
    os << " = ";
  } else {
    check_inlineable(uop);
  }

  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    os << uop->getUnaryOpType() << "(";
    if(uop->getUnaryOpType() == UnaryOpType::RandLike)
      os << "rnd";
    else
      handle(uop->in());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const BinaryOp* const bop) {
  bool istvop = isTVOp(bop);
  if (!print_inline_) {
    indent();
    os << bop->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }

    os << " = ";
  } else {
    check_inlineable(bop);
  }

  if (auto inline_bop = inline_op_str(bop->getBinaryOpType())) {
    handle(bop->lhs());
    if (istvop) {
      os << "\n";
      indent();
    }
    os << " " << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    os << bop->getBinaryOpType() << "(";
    handle(bop->lhs());
    if (istvop) {
      os << "\n";
      indent();
    }
    os << ", ";
    handle(bop->rhs());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const TernaryOp* const top) {
  bool istvop = isTVOp(top);
  if (!print_inline_) {
    indent();
    os << top->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }

    os << " = ";
  } else {
    check_inlineable(top);
  }

  os << top->getTernaryOpType() << "(";
  handle(top->in1());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in2());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in3());
  os << ")";

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const ForLoop* const fl) {
  if (fl->iter_domain()->isThread()) {
    for (auto& expr : fl->constBody().exprs())
      handle(expr);
    return;
  }

  indent();
  os << "for(size_t ";
  handle(fl->index());
  os << " = ";
  print_inline(fl->iter_domain()->start());
  os << "; ";
  handle(fl->index());
  os << " < ";
  print_inline(fl->iter_domain()->extent());
  os << "; ++";
  handle(fl->index());
  os << " ) {\n";
  indent_size++;
  for (auto& expr : fl->constBody().exprs())
    handle(expr);

  indent_size--;
  indent();
  os << "}\n";
}

void IRPrinter::handle(const IfThenElse* const ite) {
  indent();

  // IF
  os << "if ( ";
  print_inline(ite->cond());
  os << " ) { \n";

  indent_size++;
  for (auto& expr : ite->constBody().exprs()) {
    handle(expr);
  }
  indent_size--;

  // ELSE
  if (ite->hasElse()) {
    indent();
    os << "} else { \n";
    indent_size++;
    for (auto& expr : ite->constElseBody().exprs()) {
      handle(expr);
    }
    indent_size--;
  }
  indent();
  os << "}\n";
}

void IRPrinter::handle(const Allocate* const a) {
  indent();
  os << a->buf_type() << " T" << a->buffer()->name() << "[";
  print_inline(a->extent());
  os << "];" << std::endl;
}

void IRPrinter::handle(const Split* const s) {
  os << "Split: ";
  handle(s->in());
  os << " axis " << s->axis() << " by factor " << s->factor() << " -> ";
  handle(s->out());
  os << "\n";
}

void IRPrinter::handle(const Merge* const m) {
  os << "Merge: " << m->in() << " axis " << m->axis()
     << " with the following -> ";
  handle(m->out());
  os << "\n";
}

void IRPrinter::handle(const Reorder* const ro) {
  os << "Reorder: ";
  handle(ro->in());
  os << " -> ";
  handle(ro->out());
  os << "\n";
}

void IRPrinter::printKernel(
    const std::vector<Expr*>& exprs,
    const std::string& kernel_name) {
  Fusion* fusion = FusionGuard::getCurFusion();

  printHeader(fusion, kernel_name);
  for (auto* expr : exprs) {
    handle(expr);
  }
  os << "}\n";
}

std::ostream& operator<<(std::ostream& os, const Statement* const stmt) {
  IRPrinter p(os);
  p.handle(stmt);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion* f) {
  IRPrinter p(os);
  FusionGuard guard(f);
  p.handle(f);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion& f) {
  return os << &f;
}

} // namespace fuser
} // namespace jit
} // namespace torch
