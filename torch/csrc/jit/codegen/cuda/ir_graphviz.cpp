#include <torch/csrc/jit/codegen/cuda/ir_graphviz.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Private helper, generating node labels for IrGraphGenerator
class IrNodeLabel : private OptInConstDispatch {
  using DetailLevel = IrGraphGenerator::DetailLevel;

 public:
  static std::string gen(
      const Statement* node,
      DetailLevel detail_level = DetailLevel::Basic) {
    IrNodeLabel generator(detail_level);
    generator.OptInConstDispatch::handle(node);
    return generator.label_.str();
  }

 private:
  explicit IrNodeLabel(DetailLevel detail_level)
      : detail_level_(detail_level) {}

  ~IrNodeLabel() override = default;

  void handle(const Bool* b) override {
    if (b->isSymbolic()) {
      label_ << "b" << b->name();
    } else {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << "b" << b->name() << "=";
      }
      label_ << *b->value();
    }
  }

  void handle(const Float* f) override {
    if (f->isSymbolic()) {
      label_ << "f" << f->name();
    } else {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << "f" << f->name() << "=";
      }
      label_ << std::fixed << std::setprecision(2) << *f->value();
    }
  }

  void handle(const Half* h) override {
    if (h->isSymbolic()) {
      label_ << "h" << h->name();
    } else {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << "h" << h->name() << "=";
      }
      label_ << *h->value();
    }
  }

  void handle(const Int* i) override {
    if (i->isSymbolic()) {
      label_ << "i" << i->name();
    } else {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << "i" << i->name() << "=";
      }
      label_ << *i->value();
    }
  }

  void handle(const NamedScalar* ns) override {
    label_ << ns->name();
  }

  void handle(const IterDomain* id) override {
    label_ << id->getIterType();
    label_ << id->getParallelType();

    label_ << "(";
    if (!id->start()->isZeroInt()) {
      label_ << IrNodeLabel::gen(id->start()) << " : ";
    }
    label_ << IrNodeLabel::gen(id->extent());
    if (id->rawExtent() != id->extent()) {
      label_ << "\\<" << IrNodeLabel::gen(id->rawExtent()) << "\\>";
    }
    label_ << ")";
  }

  void handle(const Split* split) override {
    label_ << "Split(factor=" << IrNodeLabel::gen(split->factor()) << ")";
  }

  void handle(const Merge* merge) override {
    label_ << "Merge";
  }

 private:
  std::stringstream label_;
  const DetailLevel detail_level_;
};

} // anonymous namespace

void IrGraphGenerator::print(
    const Fusion* fusion,
    const char* filename,
    DetailLevel detail_level) {
  std::ofstream dot_file(filename);
  TORCH_CHECK(dot_file.good(), "Failed to open the IR graph file");
  dot_file << toGraphviz(fusion, detail_level);
}

std::string IrGraphGenerator::toGraphviz(
    const Fusion* fusion,
    DetailLevel detail_level) {
  IrGraphGenerator ir_graph(fusion, detail_level);
  return ir_graph.generate();
}

IrGraphGenerator::IrGraphGenerator(
    const Fusion* fusion,
    DetailLevel detail_level)
    : detail_level_(detail_level), fusion_(fusion) {
  // setup inputs & outputs
  // (indexes used to quickly check if a value is fusion input or output)
  for (const auto* input : fusion->inputs()) {
    TORCH_CHECK(inputs_.count(input) == 0);
    inputs_.insert(input);
  }
  for (const auto* output : fusion->outputs()) {
    TORCH_CHECK(outputs_.count(output) == 0);
    outputs_.insert(output);
  }
}

std::string IrGraphGenerator::getid(const Statement* stm) {
  const auto it = id_map_.find(stm);
  if (it == id_map_.end()) {
    // First reference, generate a new id
    std::stringstream new_id;
    new_id << "stm_" << next_id_++;
    id_map_.insert({stm, new_id.str()});
    return new_id.str();
  } else {
    return it->second;
  }
}

void IrGraphGenerator::addArc(
    const Statement* src,
    const Statement* dst,
    const std::string& style) {
  // We automatically visit (handle) the arc's source and destination
  handle(src);
  handle(dst);

  // generate and queue the arc definition
  std::stringstream arc_def;
  arc_def << getid(src) << " -> " << getid(dst) << " " << style;
  arcs_.push_back(arc_def.str());
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  graph_def_ << "    " << getid(expr) << " "
             << "[label=\"" << label << "\", shape=oval, color=blue, "
             << "style=filled, fillcolor=azure];\n";
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  graph_def_ << "    " << getid(val) << " [label=\"" << label
             << "\", shape=rect, color=green, fontsize=10];\n";
}

std::string IrGraphGenerator::generate() {
  // IrGraphGenerator instances are not reusable
  TORCH_CHECK(graph_def_.str().empty());
  TORCH_CHECK(visited_.empty());

  // record detail level
  graph_def_ << "// detail level: ";
  switch (detail_level_) {
    case DetailLevel::ComputeOnly:
      graph_def_ << "compute only\n";
      break;
    case DetailLevel::Basic:
      graph_def_ << "minimal\n";
      break;
    case DetailLevel::Explicit:
      graph_def_ << "explicit\n";
      break;
    case DetailLevel::Verbose:
      graph_def_ << "verbose\n";
      break;
    default:
      TORCH_CHECK(!"Unexpected detail level");
  }

  graph_def_ << "digraph fusion_ir {\n"
             << "  node [shape=circle, color=gray];\n"
             << "  edge [color=black];\n";

  // Compute graph
  generateComputeGraph();

  // Schedule graph
  if (detail_level_ > DetailLevel::ComputeOnly) {
    generateScheduleGraph();
  }

  // All expressions & values
  // (These are otherwise unreacheable (dead) nodes)
  if (detail_level_ >= DetailLevel::Verbose) {
    for (const auto* expr : fusion_->unordered_exprs()) {
      handle(expr);
    }
    for (const auto* val : fusion_->vals()) {
      handle(val);
    }
  }

  // Finally, print all arc definitions
  for (const auto& arc : arcs_) {
    graph_def_ << "  " << arc << ";\n";
  }

  graph_def_ << "}\n";

  // Make sure that all referenced nodes have been visited
  for (const auto& kv : id_map_) {
    TORCH_CHECK(visited(kv.first));
  }

  return graph_def_.str();
}

void IrGraphGenerator::generateComputeGraph() {
  graph_def_ << "  subgraph cluster_compute {\n"
             << "    label=\"compute\";\n"
             << "    style=dashed;\n";

  // Inputs
  for (const auto* input : fusion_->inputs()) {
    handle(input);
  }

  // Outputs
  for (const auto* output : fusion_->outputs()) {
    handle(output);
  }

  graph_def_ << "  }\n";
}

void IrGraphGenerator::generateScheduleGraph() {
  graph_def_ << "  subgraph cluster_schedule {\n"
             << "    label=\"schedule\";\n"
             << "    style=dashed;\n";

  // Connect TensorView with their TensorDomain
  // (this will trigger the traversal of the schedule graph)

  for (auto tv : tensor_views_) {
    addArc(tv->domain(), tv, "[style=dashed, arrowhead=none]");
    if (detail_level_ >= DetailLevel::Explicit) {
      // Maybe not the best way to handle the root domain, but should be okay
      addArc(
          tv,
          new TensorDomain(tv->getRootDomain()),
          "[style=dashed, color=green, arrowhead=none]");

      if (tv->domain()->hasRFactor())
        addArc(
            tv,
            new TensorDomain(tv->domain()->getRFactorDomain()),
            "[style=dashed, color=green, arrowhead=none]");
    }
  }

  graph_def_ << "  }\n";
}

void IrGraphGenerator::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IrGraphGenerator::handle(const Val* v) {
  if (!visited(v)) {
    visited_.insert(v);
    if (const auto* def = fusion_->origin(v)) {
      handle(def);
    }
    OptInConstDispatch::handle(v);
  }
}

void IrGraphGenerator::handle(const Expr* e) {
  if (!visited(e)) {
    visited_.insert(e);
    OptInConstDispatch::handle(e);
  }
}

void IrGraphGenerator::handle(const TensorDomain* td) {
  graph_def_ << "    " << getid(td) << " [label=\"TensorDomain\", "
             << "shape=note, color=gray, "
             << "style=filled, fillcolor=gray90, fontsize=10];\n";
  for (auto iter_domain : td->domain()) {
    addArc(iter_domain, td, "[color=gray]");
  }
}

void IrGraphGenerator::handle(const IterDomain* id) {
  graph_def_ << "    " << getid(id) << " [label=\"" << IrNodeLabel::gen(id)
             << "\", shape=cds, color=gray, fontsize=10];\n";

  if (!id->start()->isZeroInt()) {
    addArc(id->start(), id, "[color=gray]");
  }

  addArc(id->rawExtent(), id, "[color=gray]");

  if (detail_level_ >= DetailLevel::Explicit &&
      id->rawExtent() != id->extent()) {
    addArc(id->extent(), id, "[color=gray, style=dashed]");
  }
}

void IrGraphGenerator::handle(const Bool* b) {
  printValue(b, IrNodeLabel::gen(b, detail_level_));
}

void IrGraphGenerator::handle(const Float* f) {
  printValue(f, IrNodeLabel::gen(f, detail_level_));
}

void IrGraphGenerator::handle(const Half* h) {
  printValue(h, IrNodeLabel::gen(h, detail_level_));
}

void IrGraphGenerator::handle(const Int* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
}

void IrGraphGenerator::handle(const TensorView* tv) {
  std::stringstream label;
  label << "{T" << tv->name() << "|";
  label << "{";
  bool first_axis = true;
  for (auto iter_domain : tv->domain()->domain()) {
    if (first_axis) {
      first_axis = false;
    } else {
      label << "|";
    }
    label << IrNodeLabel::gen(iter_domain);
  }
  label << "}}";

  const bool is_input = inputs_.find(tv) != inputs_.end();
  const bool is_output = outputs_.find(tv) != outputs_.end();

  const char* style = is_input ? "style=filled, fillcolor=palegreen"
      : is_output              ? "style=filled, fillcolor=lightblue"
                               : "style=filled, fillcolor=beige";

  graph_def_ << "    " << getid(tv) << " [label=\"" << label.str()
             << "\", shape=Mrecord, color=brown, " << style << "];\n";

  if (const auto* compute_at_view = tv->getComputeAtView()) {
    std::stringstream arc_style;
    arc_style << "[color=red, style=dashed, label=\""
              << "ComputeAt(" << tv->getRelativeComputeAtAxis() << ")\"]";
    addArc(tv, compute_at_view, arc_style.str());
  }

  tensor_views_.push_back(tv);
}

void IrGraphGenerator::handle(const UnaryOp* uop) {
  // node
  std::stringstream label;
  label << uop->getUnaryOpType();
  printExpr(uop, label.str());

  // inputs & outputs
  addArc(uop->in(), uop);
  addArc(uop, uop->out());
}

void IrGraphGenerator::handle(const BinaryOp* bop) {
  // node
  std::stringstream label;
  label << bop->getBinaryOpType();
  printExpr(bop, label.str());

  // inputs & outputs
  addArc(bop->lhs(), bop);
  addArc(bop->rhs(), bop, "[color=blue]");
  addArc(bop, bop->out());
}

void IrGraphGenerator::handle(const TernaryOp* op) {
  // node
  std::stringstream label;
  label << op->getTernaryOpType();
  printExpr(op, label.str());

  // inputs & outputs
  addArc(op->in1(), op);
  addArc(op->in2(), op, "[color=blue]");
  addArc(op->in3(), op, "[color=brown]");
  addArc(op, op->out());
}

void IrGraphGenerator::handle(const BroadcastOp* op) {
  printExpr(op, "Broadcast");
  addArc(op->in(), op);
  addArc(op, op->out());
}

void IrGraphGenerator::handle(const ReductionOp* op) {
  // node
  std::stringstream label;
  label << "Reduction(" << op->getReductionOpType() << ")";
  printExpr(op, label.str());

  // inputs & outputs
  addArc(op->in(), op);
  addArc(op->init(), op, "[color=blue]");
  addArc(op, op->out());
}

void IrGraphGenerator::handle(const Split* split) {
  printExpr(split, IrNodeLabel::gen(split));
  addArc(split->in(), split);
  addArc(split, split->outer());
  addArc(split, split->inner());
}

void IrGraphGenerator::handle(const Merge* merge) {
  printExpr(merge, IrNodeLabel::gen(merge));
  addArc(merge->outer(), merge);
  addArc(merge->inner(), merge);
  addArc(merge, merge->out());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
