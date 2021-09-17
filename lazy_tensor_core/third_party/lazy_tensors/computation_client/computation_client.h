#ifndef COMPUTATION_CLIENT_COMPUTATION_CLIENT_H_
#define COMPUTATION_CLIENT_COMPUTATION_CLIENT_H_

#include <c10/util/Optional.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lazy_tensors/computation_client/client_data.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/types.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/status.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

class GenericComputation {
 public:
  virtual StatusOr<ProgramShape> GetProgramShape() const = 0;

  virtual ~GenericComputation() = default;
};

class ComputationClient {
 public:
  class Computation {
   public:
    Computation(std::shared_ptr<GenericComputation> computation,
                ProgramShape program_shape, std::vector<std::string> devices)
        : computation_(std::move(computation)),
          program_shape_(std::move(program_shape)),
          devices_(std::move(devices)) {}

    virtual ~Computation() {}

    GenericComputation* computation() const { return computation_.get(); }

    const ProgramShape& program_shape() const { return program_shape_; }

    const std::vector<std::string>& devices() const { return devices_; }

   private:
    std::shared_ptr<GenericComputation> computation_;
    ProgramShape program_shape_;
    std::vector<std::string> devices_;
  };

  using ComputationPtr = std::shared_ptr<Computation>;

  struct CompileInstance {
    CompileInstance() = default;
    CompileInstance(std::shared_ptr<GenericComputation> computation,
                    std::string compilation_device,
                    std::vector<std::string> devices, const Shape* output_shape)
        : computation(std::move(computation)),
          compilation_device(std::move(compilation_device)),
          devices(std::move(devices)),
          output_shape(output_shape) {}

    std::shared_ptr<GenericComputation> computation;
    std::string compilation_device;
    std::vector<std::string> devices;
    const Shape* output_shape = nullptr;
  };

  struct ExecuteOptions {
    bool explode_tuple = true;
  };

  struct ExecuteComputationOptions : public ExecuteOptions {};

  struct ExecuteReplicatedOptions : public ExecuteOptions {};

  struct ExecuteParallelOptions : public ExecuteOptions {};

  using DataPtr = lazy_tensors::client::DataPtr;
  using TensorSource = lazy_tensors::client::TensorSource;

  // Describes an operation to be fed to the ExecuteChained() API.
  // If the device_data member is not nullptr, this operation is a device data
  // input. Otherwise computation must not be nullptr, and represents the
  // computation to be executed. The indices of the inputs to the computation,
  // are coming from the inputs member. Since the operations fed to
  // ExecuteChained() are a valid post-order, the op_index indices listed within
  // the inputs member must be lower of the index of the current
  // ExecuteChainedOp within the post-order. If the outputs member has values,
  // the result of this ExecuteChainedOp will become an output of the
  // ExecuteChained() API, with the output_index output of this ExecuteChainedOp
  // feeding the result_index result.
  struct ExecuteChainedOp {
    struct Input {
      size_t op_index;
      c10::optional<size_t> output_index;
    };
    struct Output {
      size_t result_index;
      c10::optional<size_t> output_index;
    };

    DataPtr device_data;
    ComputationPtr computation;
    std::vector<Output> outputs;
    std::vector<Input> inputs;
  };

  struct MemoryInfo {
    int64 kb_free = 0;
    int64 kb_total = 0;
  };

  static std::unique_ptr<ComputationClient> Create();

  virtual ~ComputationClient() {}

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fashion.
  virtual DataPtr CreateDataPlaceholder(std::string device, Shape shape) = 0;

  // Transfers local tensor values to the TPU servers and fetches the handles.
  virtual std::vector<DataPtr> TransferToServer(
      lazy_tensors::Span<const TensorSource> tensors) = 0;

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  virtual std::vector<Literal> TransferFromServer(
      lazy_tensors::Span<const DataPtr> handles) = 0;

  // Compiles a set of computations.
  virtual std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) = 0;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  virtual std::vector<DataPtr> ExecuteComputation(
      const Computation& computation,
      lazy_tensors::Span<const DataPtr> arguments, const std::string& device,
      const ExecuteComputationOptions& options) = 0;

  // Executes the computation in replicated mode.
  // The size of the arguments vector is the number of replicas to execute,
  // and it must match the size of the computation.devices() as well as the
  // devices passed as argument. The destination devices for each replicated
  // computation come from the devices the Data objects are stored into, which
  // must match the devices argument. Within arguments[i], every Data
  // object must be coming from the same device. Returns a vector (of the same
  // size of the arguments vector) with the results of the parallel execution.
  // The result[i], a vector itself, will be the result of the computation fed
  // with arguments[i]. If options.explode_tuple is true, the output tuples will
  // be decomposed into their single elements.
  virtual std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) = 0;

  // Executes the computations in parallel. Each computation must target a
  // different device, and the the common device of arguments[i] must match
  // devices[i]. The computations[i] computation is fed with arguments[i]
  // arguments.
  // Returns a vector of vectors of device side Data object, with result[i]
  // being the return value of computations[i]. If options.explode_tuple is
  // true, the output tuples will be decomposed into their single elements.
  virtual std::vector<std::vector<DataPtr>> ExecuteParallel(
      lazy_tensors::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteParallelOptions& options) = 0;

  // Executes a serie of operations, whose results are input of other
  // operations. The ops is a valid post-order for the execution, which means
  // that the inputs of op at index I, will have to be coming from ops at index
  // lower than I. It returns a vector of device data shared pointers, one for
  // every ExecuteChainedOp marked with is_result=true, in the order they appear
  // within the ops post-order.
  virtual std::vector<DataPtr> ExecuteChained(
      lazy_tensors::Span<const ExecuteChainedOp> ops,
      const std::string& device) = 0;

  virtual std::vector<std::vector<DataPtr>> DeconstructTuple(
      lazy_tensors::Span<const DataPtr> tuples) = 0;

  // Returns a unique string which identifies the resource domain of a given
  // device. Within a resource domain, handles to device memory or compiled
  // computations can be used for all devices part of such domain.
  virtual std::string GetResourceDomain(const std::string& device) const = 0;

  virtual std::string GetDefaultDevice() const = 0;

  virtual size_t GetNumDevices() const = 0;

  virtual std::vector<std::string> GetLocalDevices() const = 0;

  virtual std::vector<std::string> GetAllDevices() const = 0;

  virtual void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) = 0;

  virtual std::shared_ptr<std::vector<std::string>> GetReplicationDevices() = 0;

  virtual void SetRngSeed(size_t seed) = 0;

  virtual std::map<std::string, Metric> GetMetrics() const = 0;

  virtual MemoryInfo GetMemoryInfo(const std::string& device) = 0;

  virtual void PrepareToExit() = 0;

  // Retrieves the set of devices to be passed to the computation client
  // Compile() API. If the devices array is empty, a vector with the single
  // device will be returned. Otherwise a vector with the devices content will
  // be returned.
  std::vector<std::string> GetCompilationDevices(
      const std::string& device, lazy_tensors::Span<const std::string> devices);

  // Retrieves the ordinal number out of a device string. This is the number
  // after the last ':' character of the device string.
  static int64 GetDeviceOrdinal(const std::string& device);

  // Returns the ComputationClient singleton.
  static ComputationClient* Get();

  static ComputationClient* GetIfInitialized();

 protected:
  // Metrics common to all client interfaces.
  static metrics::Metric* TransferToServerMetric();
  static metrics::Metric* TransferToServerTransformMetric();
  static metrics::Metric* TransferFromServerMetric();
  static metrics::Metric* CompileMetric();
  static metrics::Metric* ExecuteMetric();
  static metrics::Metric* ExecuteReplicatedMetric();
  static metrics::Metric* ExecuteParallelMetric();
  static metrics::Metric* ExecuteChainedMetric();
  static metrics::Metric* DeconstructTupleMetric();
  static metrics::Counter* CreateDataHandlesCounter();
  static metrics::Counter* ReleaseDataHandlesCounter();
  static metrics::Counter* DestroyDataHandlesCounter();
  static metrics::Metric* ReleaseDataHandlesTimeMetric();
  static metrics::Counter* CreateCompileHandlesCounter();
  static metrics::Counter* ReleaseCompileHandlesCounter();
  static metrics::Counter* DestroyCompileHandlesCounter();
  static metrics::Metric* ReleaseCompileHandlesTimeMetric();
  static metrics::Metric* InboundDataMetric();
  static metrics::Metric* OutboundDataMetric();
};

at::Tensor MakeTensorFromComputationData(
    const ComputationClient::DataPtr data,
    c10::optional<at::ScalarType> logical_scalar_type = c10::nullopt);

ComputationClient::DataPtr MakeComputationDataFromTensor(
    const at::Tensor& tensor, const Shape& shape, const std::string& device);

}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_COMPUTATION_CLIENT_H_
