#pragma once

#include "lazy_tensors/computation_client/computation_client.h"

namespace lazy_tensors {
namespace compiler {

class TSComputationClient : public ComputationClient {
  using Data = client::Data;

 public:
  struct TSData : public Data {
    TSData(const at::Tensor& data, client::ShapeData shape, std::string device)
        : Data(std::move(device), std::move(shape)), data_(data) {}

    TSData(client::ShapeData shape, std::string device)
        : Data(std::move(device), std::move(shape)) {}

    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<int64>(this);
    }

    void Assign(const Data& data) override {
      data_ = static_cast<const TSData&>(data).data_;
    }

    bool HasValue() const override { return data_.defined(); }

    at::Tensor data_;
  };

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      lazy_tensors::Span<const TensorSource> tensors) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<Literal> TransferFromServer(
      lazy_tensors::Span<const DataPtr> handles) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation,
      lazy_tensors::Span<const DataPtr> arguments, const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      lazy_tensors::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<DataPtr> ExecuteChained(
      lazy_tensors::Span<const ExecuteChainedOp> ops,
      const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      lazy_tensors::Span<const DataPtr> tuples) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override { return 1; }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void SetRngSeed(size_t seed) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::map<std::string, Metric> GetMetrics() const override { return {}; }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;

  static at::DeviceType HardwareDeviceType();
};

lazy_tensors::ComputationClient* TSClientGet();

lazy_tensors::ComputationClient* TSClientGetIfInitialized();

}  // namespace compiler
}  // namespace lazy_tensors
