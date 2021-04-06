#include <torch/csrc/distributed/spmd/event_impl.h>

#include <torch/types.h>

namespace torch {
namespace distributed {
namespace spmd {

namespace {

// NB: need to call torch::class_ to register event classes in the map returned
// by c10::getCustomClassTypeMap(). Otherwise, these classes cannot be wrapped
// by an IValue.
// NB: add this line here instead of in spmd/init.cpp because these classes are
// not yet meant to be visible from Python.
static const auto prepareModuleEvent =
    torch::class_<PrepareModuleEvent>("spmd", "_PrepareModuleEvent");
static const auto preForwardEvent =
    torch::class_<PreForwardEvent>("spmd", "_PreForwardEvent");
static const auto localgradReadyEvent =
    torch::class_<LocalGradReadyEvent>("spmd", "_LocalGradReadyEvent");
static const auto bucketReadyEvent =
    torch::class_<BucketReadyEvent>("spmd", "_BucketReadyEvent");
static const auto commDoneEvent =
    torch::class_<CommDoneEvent>("spmd", "_CommDoneEvent");
static const auto globalGradReadyEvent =
    torch::class_<GlobalGradReadyEvent>("spmd", "_GlobalGradReadyEvent");

} // namespace

} // namespace spmd
} // namespace distributed
} // namespace torch
