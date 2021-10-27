#pragma once

#include "lazy_tensors/computation_client/computation_client.h"

namespace lazy_tensors {
namespace compiler {

class TSComputationClient : public ComputationClient {
  using Data = client::Data;

 public:
  class TSData : public Data {
    public:

      TSData(const at::Tensor& data, client::ShapeData shape, std::string device)
          : Data(std::move(device), std::move(shape)), data_(data) {}

      TSData(client::ShapeData shape, std::string device)
          : Data(std::move(device), std::move(shape)) {}

      OpaqueHandle GetOpaqueHandle() override {
        return reinterpret_cast<int64_t>(this);
      }

      void Assign(const Data& data) override {
        data_ = static_cast<const TSData&>(data).data_;
      }

      bool HasValue() const override { return data_.defined(); }

      at::Tensor data() { return data_; }

    private:
      at::Tensor data_;
  };

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      c10::ArrayRef<TensorSource> tensors) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<Literal> TransferFromServer(
      c10::ArrayRef<DataPtr> handles) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, c10::ArrayRef<DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      c10::ArrayRef<std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      c10::ArrayRef<Computation*> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      c10::ArrayRef<std::string> devices,
      const ExecuteParallelOptions& options) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<DataPtr> ExecuteChained(c10::ArrayRef<ExecuteChainedOp> ops,
                                      const std::string& device) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      c10::ArrayRef<DataPtr> tuples) override {
    LOG(FATAL) << "Not implemented yet.";
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
    LOG(FATAL) << "Not implemented yet.";
  }

  std::map<std::string, Metric> GetMetrics() const override { return {}; }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;

  static at::DeviceType HardwareDeviceType();
};

lazy_tensors::ComputationClient* TSClientGet();

lazy_tensors::ComputationClient* TSClientGetIfInitialized();

}  // namespace compiler
}  // namespace lazy_tensors
