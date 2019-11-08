#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

// Directly after tracing, we have an ill-formed graph with blocks inserted.
// Example:
//
// graph(%self : ClassType<Module>,
//       %input.1 : Float(3, 4)):
//   %1 : ClassType<Module> = prim::GetAttr[name="relu1"](%self)
//   %2 : ClassType<Module> = prim::GetAttr[name="relu2"](%self)
//   %3 : ClassType<Module> = prim::GetAttr[name="rrr"](%2)
//    = prim::TracedModuleForward[scope="__module.relu1"]()
//     block0():
//       %input : Float(3, 4) = aten::relu(%input.1),
//       -> ()
//    = prim::TracedModuleForward[scope="__module.relu2"](),
//     block0():
//        = prim::TracedModuleForward[scope="__module.relu2.rrr"](),
//         block0():
//           %6 : Float(3, 4) = aten::relu(%input),
//           -> ()
//       -> ()
//   return (%6)
//
// In this pass, we:
//   1) Lift Value defs to as high of a scope as needed to ensure that
//      they dominate all their uses. For example, `input` in the above
//      graph needs to be lifted to the top-level block so that its use
//      in the second `relu` operator is dominated.
//   2) Lambda lift the blocks. This ensures that all values used within
//      each scope have their defs captured.
//   3) Convert the scope blocks into methods on their respective Modules,
//      and convert TracedModuleForward nodes to CallMethod nodes into those
//      methods.
//
//  Then, we'll have a well-formed graph with proper method calls.
TORCH_API void FixupTraceScopeBlocks(
    std::shared_ptr<Graph>& graph,
    script::Module* self);

} // namespace jit
} // namespace torch
