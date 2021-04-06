#include <torch/csrc/distributed/spmd/event.h>

#include <torch/types.h>

namespace torch {
namespace distributed {
namespace spmd {

namespace {

// NB: need to call torch::class_ to register Event in the map returned by
// c10::getCustomClassTypeMap(). Otherwise, Event cannot be wrapped within
// an IValue.
// NB: add this line here instead of in spmd/init.cpp because Event is not
// yet meant to be visible from Python.
static const auto event = torch::class_<Event>("spmd", "_Event");

} // namespace

} // namespace spmd
} // namespace distributed
} // namespace torch
