#include "program_facts.h"

#include <algorithm>
#include <stdexcept>
#include <variant>

namespace torch::inductor::loops_ir::op_options {

namespace {

RootRef resolveChain(const std::string& name,
                     const std::map<std::string, const View*>& views,
                     std::set<std::string>& seen) {
  auto it = views.find(name);
  if (it == views.end()) {
    return RootRef{name, 0};
  }
  if (!seen.insert(name).second) {
    throw std::invalid_argument("analyzeProgram: view alias cycle at '" + name + "'");
  }
  RootRef src = resolveChain(it->second->source, views, seen);
  src.offset += it->second->offset;
  return src;
}

void addUse(ProgramFacts& f, const std::string& name, size_t node) {
  auto& u = f.users[f.resolve(name).root];
  if (u.empty() || u.back() != node) {
    u.push_back(node);
  }
}

void addRoot(ProgramFacts& f, const Buffer& b, std::optional<size_t> node) {
  if (f.roots.count(b.name)) {
    throw std::invalid_argument("analyzeProgram: duplicate buffer '" + b.name + "'");
  }
  f.roots[b.name] = RootRef{b.name, 0};
  f.root_elements[b.name] = b.elements;
  f.root_dtype[b.name] = b.dtype;
  if (node) {
    f.producer[b.name] = *node;
  }
}

} // namespace

const RootRef& ProgramFacts::resolve(const std::string& name) const {
  auto it = roots.find(name);
  if (it == roots.end()) {
    throw std::invalid_argument("analyzeProgram: unknown buffer or view '" + name + "'");
  }
  return it->second;
}

bool ProgramFacts::isKernelFunction(const KernelProgram& p, size_t node) const {
  return std::holds_alternative<KernelFunction>(p.nodes().at(node));
}

ProgramFacts analyzeProgram(const KernelProgram& program) {
  ProgramFacts f;
  for (const Buffer& b : program.inputs()) {
    addRoot(f, b, std::nullopt);
  }
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    const auto& node = program.nodes()[i];
    if (const auto* kf = std::get_if<KernelFunction>(&node)) {
      for (const Compute& c : kf->regions()) {
        for (const Buffer& b : c.results()) {
          addRoot(f, b, i);
        }
      }
    } else {
      for (const Buffer& b : std::get<ExternCall>(node).results()) {
        addRoot(f, b, i);
      }
    }
  }
  std::map<std::string, const View*> views;
  for (const View& v : program.views()) {
    views[v.name] = &v;
  }
  for (const View& v : program.views()) {
    std::set<std::string> seen;
    RootRef r = resolveChain(v.name, views, seen);
    if (!f.root_elements.count(r.root)) {
      throw std::invalid_argument("analyzeProgram: view '" + v.name + "' has no storage root");
    }
    f.roots[v.name] = r;
  }
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    const auto& node = program.nodes()[i];
    if (const auto* kf = std::get_if<KernelFunction>(&node)) {
      for (const Compute& c : kf->regions()) {
        for (const std::string& in : c.inputs()) {
          addUse(f, in, i);
        }
      }
    } else {
      for (const std::string& in : std::get<ExternCall>(node).inputs()) {
        addUse(f, in, i);
      }
    }
  }
  for (const Descriptor& d : program.outputs()) {
    f.escaping_roots.insert(f.resolve(d.buffer).root);
  }
  return f;
}

} // namespace torch::inductor::loops_ir::op_options
