#include <ir_graphviz.h>

#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <type.h>

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

  void handle(const Double* d) override {
    if (d->isSymbolic()) {
      label_ << "d" << d->name();
    } else {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << "d" << d->name() << "=";
      }
      label_ << *d->value();
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
    label_ << ")";
  }

  void handle(const Split* split) override {
    label_ << "Split(inner=" << (split->innerSplit() ? "true" : "false")
           << ", factor=" << IrNodeLabel::gen(split->factor()) << ")";
  }

  void handle(const Merge* merge) override {
    label_ << "Merge";
  }

 private:
  std::stringstream label_;
  const DetailLevel detail_level_;
};

// Small color palette from the X11 theme
static const char* getColorFromIndex(size_t index) {
  const size_t number_of_colors = 10;
  index = index % number_of_colors;
  switch (index) {
    case 0: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "azure";
    case 1: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "pink";
    case 2: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "green";
    case 3: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "grey";
    case 4: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "yellow";
    case 5: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "lavender";
    case 6: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "cyan";
    case 7: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "white";
    case 8: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "magenta";
    case 9: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "red";
    default:
      break;
  }
  return "";
}

} // anonymous namespace

void IrGraphGenerator::print(
    const Fusion* fusion,
    const char* filename,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map) {
  std::ofstream dot_file(filename);
  TORCH_CHECK(dot_file.good(), "Failed to open the IR graph file");
  dot_file << toGraphviz(fusion, detail_level, expr_color_map);
}

std::string IrGraphGenerator::toGraphviz(
    const Fusion* fusion,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map) {
  IrGraphGenerator ir_graph(fusion, detail_level, expr_color_map);
  return ir_graph.generate();
}

IrGraphGenerator::IrGraphGenerator(
    const Fusion* fusion,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map)
    : detail_level_(detail_level),
      fusion_(fusion),
      expr_color_map_(expr_color_map) {
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
             << "style=filled, fillcolor=";
  if (expr_color_map_ != nullptr && expr_color_map_->count(expr)) {
    graph_def_ << getColorFromIndex(expr_color_map_->at(expr));
  } else {
    graph_def_ << "azure";
  }
  graph_def_ << "];\n";
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
          IrBuilder::create<TensorDomain>(tv->getRootDomain()),
          "[style=dashed, color=green, arrowhead=none]");

      if (tv->domain()->hasRFactor())
        addArc(
            tv,
            IrBuilder::create<TensorDomain>(tv->domain()->getRFactorDomain()),
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
    if (const auto* def = v->definition()) {
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

  addArc(id->extent(), id, "[color=gray]");
}

void IrGraphGenerator::handle(const Bool* b) {
  printValue(b, IrNodeLabel::gen(b, detail_level_));
}

void IrGraphGenerator::handle(const Double* d) {
  printValue(d, IrNodeLabel::gen(d, detail_level_));
}

void IrGraphGenerator::handle(const Int* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
}

void IrGraphGenerator::handle(const ComplexDouble* i) {
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

  tensor_views_.push_back(tv);
}

void IrGraphGenerator::handle(const FullOp* fop) {
  // node
  printExpr(fop, "full");

  // inputs & outputs
  addArc(fop->getFillValue(), fop);
  addArc(fop, fop->output(0));
}

void IrGraphGenerator::handle(const ARangeOp* aop) {
  // node
  printExpr(aop, "arange");

  // inputs & outputs
  addArc(aop->start(), aop);
  addArc(aop->end(), aop);
  addArc(aop->step(), aop);
  addArc(aop, aop->output(0));
}

void IrGraphGenerator::handle(const EyeOp* eop) {
  // node
  printExpr(eop, "eye");

  // inputs & outputs
  addArc(eop, eop->output(0));
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

void IrGraphGenerator::handle(const RNGOp* op) {
  // node
  std::stringstream label;
  label << op->getRNGOpType();
  printExpr(op, label.str());

  // inputs & outputs
  addArc(op, op->output(0));
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
