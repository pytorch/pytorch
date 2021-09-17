#include "lazy_tensor_core/csrc/op_by_op_executor.h"

#include <list>
#include <unordered_map>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_cat.h"

namespace torch_lazy_tensors {
namespace {

c10::optional<size_t> GetOutputIndex(bool is_device_data, size_t index) {
  // The output of every result of an op-by-op computation is wrapped into a
  // tuple, so we need to use the index to extract it. Device data instead is
  // already unwrapped, so we need to pass an empty index so that TF/XRT code
  // uses the result buffer directly.
  if (is_device_data) {
    return c10::nullopt;
  }
  return index;
}

const lazy_tensors::Shape& GetParameterShape(
    const ir::Output& operand, const lazy_tensors::Shape& input_shape) {
  // See comment in GetOutputIndex() about device data WRT computation outpout
  // shape handling.
  const ir::ops::DeviceData* device_data =
      ir::ops::DeviceData::Cast(operand.node);
  return device_data != nullptr ? input_shape
                                : lazy_tensors::ShapeUtil::GetTupleElementShape(
                                      input_shape, operand.index);
}

lazy_tensors::hash_t ComputeNodeKey(
    const ir::Node* node,
    lazy_tensors::Span<const lazy_tensors::Shape* const> input_shapes,
    const lazy_tensors::hash_t& seed) {
  lazy_tensors::hash_t key = seed;
  const auto& operands = node->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    key = lazy_tensors::util::HashCombine(
        key, lazy_tensors::util::ShapeHash(
                 GetParameterShape(operands[i], *input_shapes[i])));
  }
  key = lazy_tensors::util::HashCombine(
      key, lazy_tensors::util::ShapeHash(node->shape()));
  return lazy_tensors::util::HashCombine(key, node->node_hash());
}

std::shared_ptr<lazy_tensors::GenericComputation> BuildNodeComputation(
    const ir::Node* node,
    lazy_tensors::Span<const lazy_tensors::Shape* const> input_shapes,
    const Device& device) {
  auto loctx = ir::LoweringContext::Create("BuildNodeComputation", device);
  const auto& operands = node->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    loctx->AddParameter(operands[i], i, *input_shapes[i],
                        lazy_tensors::StrCat("p", i));
  }
  loctx->LowerNodeToResult(node);
  return ConsumeValue(loctx->Build());
}

lazy_tensors::hash_t GetNodesKeySeed(
    const std::string& device, lazy_tensors::Span<const std::string> devices) {
  return lazy_tensors::util::MHash(device, devices);
}

}  // namespace

OpByOpExecutor::OpByOpExecutor(size_t compile_cache_size)
    : compile_cache_(compile_cache_size) {}

std::vector<lazy_tensors::ComputationClient::ExecuteChainedOp>
OpByOpExecutor::BuildOps(lazy_tensors::Span<const ir::Value> roots,
                         const std::string& device,
                         lazy_tensors::Span<const std::string> devices) {
  std::vector<const ir::Node*> root_nodes;
  root_nodes.reserve(roots.size());
  for (auto& root : roots) {
    root_nodes.push_back(root.node.get());
  }
  std::vector<const ir::Node*> post_order =
      ir::Util::ComputePostOrder(root_nodes);
  LTC_VALUE_METRIC("OpByOpGraphSize", post_order.size());
  LTC_VLOG(5) << "TensorsGraphSize=" << post_order.size();

  std::unordered_map<const ir::Node*, size_t> node_to_index;
  node_to_index.reserve(post_order.size());
  for (size_t i = 0; i < post_order.size(); ++i) {
    node_to_index[post_order[i]] = i;
  }

  auto compilation_devices =
      lazy_tensors::ComputationClient::Get()->GetCompilationDevices(device,
                                                                    devices);
  lazy_tensors::hash_t nodes_key_seed =
      GetNodesKeySeed(device, compilation_devices);
  Device exec_device(device);
  std::vector<lazy_tensors::hash_t> cache_keys;
  std::unordered_map<lazy_tensors::hash_t, std::vector<size_t>,
                     lazy_tensors::util::HashReducer>
      compile_indices;
  std::unordered_map<lazy_tensors::hash_t, size_t,
                     lazy_tensors::util::HashReducer>
      cache_keys_instance;
  std::list<lazy_tensors::Shape> compile_shapes;
  std::vector<bool> device_data_ops(post_order.size());
  std::vector<lazy_tensors::Shape> ops_shapes(post_order.size());
  std::vector<lazy_tensors::ComputationClient::CompileInstance>
      compile_instances;
  std::vector<lazy_tensors::ComputationClient::ExecuteChainedOp>
      chained_exec_ops(post_order.size());
  for (size_t i = 0; i < post_order.size(); ++i) {
    const ir::Node* node = post_order[i];
    lazy_tensors::ComputationClient::ExecuteChainedOp& cxop =
        chained_exec_ops[i];
    const ir::ops::DeviceData* device_data = ir::ops::DeviceData::Cast(node);
    if (device_data != nullptr) {
      cxop.device_data = device_data->data();
      ops_shapes[i] = cxop.device_data->shape();
      device_data_ops[i] = true;
    } else {
      std::vector<const lazy_tensors::Shape*> op_input_shapes;
      for (auto& operand : node->operands()) {
        size_t op_index = node_to_index.at(operand.node);
        cxop.inputs.push_back(
            {op_index,
             GetOutputIndex(device_data_ops[op_index], operand.index)});
        op_input_shapes.push_back(&ops_shapes[op_index]);
      }

      lazy_tensors::hash_t cache_key =
          ComputeNodeKey(node, op_input_shapes, nodes_key_seed);
      cxop.computation = compile_cache_.Get(cache_key);
      if (cxop.computation == nullptr) {
        LTC_COUNTER("OpByOpCompileCacheMiss", 1);

        // Within a single IR graph, there can be many duplicated IR nodes, so
        // make sure we do not issue a compilation for each one of those.
        auto& cache_key_indices = compile_indices[cache_key];
        cache_key_indices.push_back(i);
        if (cache_key_indices.size() == 1) {
          cache_keys.push_back(cache_key);
          cache_keys_instance[cache_key] = compile_instances.size();

          auto computation =
              BuildNodeComputation(node, op_input_shapes, exec_device);
          lazy_tensors::ProgramShape program_shape =
              ConsumeValue(computation->GetProgramShape());
          compile_shapes.push_back(MakeShapeWithDeviceLayout(
              program_shape.result(), exec_device.hw_type));
          compile_instances.push_back({std::move(computation), device,
                                       compilation_devices,
                                       &compile_shapes.back()});
          ops_shapes[i] = compile_shapes.back();
        } else {
          ops_shapes[i] = *compile_instances[cache_keys_instance.at(cache_key)]
                               .output_shape;
        }
      } else {
        ops_shapes[i] = cxop.computation->program_shape().result();
      }
    }
  }
  // Fixup the requested outputs (roots) within the chained ops vector.
  for (size_t i = 0; i < roots.size(); ++i) {
    size_t op_index = node_to_index.at(roots[i].node.get());
    chained_exec_ops[op_index].outputs.push_back(
        {i, GetOutputIndex(device_data_ops[op_index], roots[i].index)});
  }

  // If we missed the cache for certain ops, compile them now and fixup the
  // chained ops vector.
  if (!compile_instances.empty()) {
    LTC_VLOG(3) << "Compiling " << compile_instances.size()
                << " computations on device " << device;
    auto computation_ptrs = lazy_tensors::ComputationClient::Get()->Compile(
        std::move(compile_instances));
    LTC_VLOG(3) << "Compiling " << computation_ptrs.size()
                << " computations on device " << device << " done!";
    for (size_t i = 0; i < computation_ptrs.size(); ++i) {
      compile_cache_.Add(cache_keys[i], computation_ptrs[i]);
      for (auto index : compile_indices[cache_keys[i]]) {
        chained_exec_ops[index].computation = computation_ptrs[i];
      }
    }
  }
  return chained_exec_ops;
}

std::vector<lazy_tensors::ComputationClient::DataPtr> OpByOpExecutor::Execute(
    lazy_tensors::Span<const ir::Value> roots, const std::string& device,
    lazy_tensors::Span<const std::string> devices) {
  auto chained_exec_ops = BuildOps(roots, device, devices);
  return lazy_tensors::ComputationClient::Get()->ExecuteChained(
      chained_exec_ops, device);
}

OpByOpExecutor::AsyncTask OpByOpExecutor::ExecuteAsync(
    lazy_tensors::Span<const ir::Value> roots, const std::string& device,
    lazy_tensors::Span<const std::string> devices) {
  std::vector<ir::Value> roots_vector(roots.begin(), roots.end());
  std::vector<std::string> devices_vector(devices.begin(), devices.end());
  auto taskfn = [this, roots = std::move(roots_vector),
                 devices = std::move(devices_vector), device]() -> AsyncResult {
    return Execute(roots, device, devices);
  };

  AsyncTask async = AsyncTask(std::move(taskfn));
  return async.Schedule();
}

OpByOpExecutor* OpByOpExecutor::Get() {
  static const lazy_tensors::int64 compile_cache_size =
      lazy_tensors::sys_util::GetEnvInt("SPLIT_EXECUTOR_CACHE_SIZE", 2048);
  static OpByOpExecutor* split_executor =
      new OpByOpExecutor(compile_cache_size);
  return split_executor;
}

}  // namespace torch_lazy_tensors
