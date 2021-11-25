#include <aten/src/ATen/TracerMode.h>
#include <c10/core/HugePagesAllocator.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/types.h> // torch::tensor
#include <iostream>
#include <regex>

#include <torch/csrc/jit/passes/freeze_module.h>
//#include <torch/csrc/jit/passes/memory_planning.h>

#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <torch/csrc/jit/passes/remove_mutation.h> // RemoveMutation

#include <torch/csrc/jit/ir/irparser.h> // for parseIR

#include <torch/csrc/jit/ir/alias_analysis.h>

#include <torch/csrc/jit/passes/tensorexpr_fuser.h> // RemoveProfileNodesAndSpecializeTypes

#include <torch/csrc/jit/jit_log.h> // getHeader

#include <torch/csrc/jit/serialization/import.h> // module load

#include <torch/csrc/autograd/profiler_kineto.h>
//#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>

#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

#include <torch/csrc/jit/passes/inliner.h>

#include <jemalloc/jemalloc.h>

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

void write_cb(void* opaque, const char* to_write) {
  auto* arg = (std::ofstream*)opaque;
  size_t bytes = strlen(to_write);
  arg->write(to_write, bytes);
}

void je_malloc_print_stats(const std::string& fp) {
  try {
    std::ofstream text_stream;
    text_stream.open(fp + ".txt");
    jemalloc_stats_print(write_cb, (void*)&text_stream, "");
    std::ofstream json_stream;
    json_stream.open(fp + ".json");
    jemalloc_stats_print(write_cb, (void*)&json_stream, "J");
  } catch (const std::exception& e) {
    std::cout << e.what() << "\n";
  }
}

using namespace torch::jit;

inline std::string ReplaceString(
    std::string subject,
    const std::string& search,
    const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
};

std::pair<std::shared_ptr<Graph>, torch::jit::script::Module>
get_graph_from_model(std::string pt_path) {
  torch::jit::script::Module module;

  module = torch::jit::load(pt_path);
  std::shared_ptr<Module> module_ptr;
  module.eval();
  module_ptr = std::make_shared<Module>(freeze_module(module));
  auto forward = module_ptr->get_method("forward");
  auto graph = forward.graph();
  graph->eraseInput(0);
  for (const auto& inp : graph->inputs()) {
    auto name = inp->debugName();
    if (name.find('.') != std::string::npos)
      inp->setDebugName(ReplaceString(name, ".", "_"));
  }
  for (const auto& node : graph->nodes()) {
    for (const auto& inp : node->inputs()) {
      auto name = inp->debugName();
      if (name.find('.') != std::string::npos)
        inp->setDebugName(ReplaceString(name, ".", "_"));
    }
    for (const auto& inp : node->outputs()) {
      auto name = inp->debugName();
      if (name.find('.') != std::string::npos)
        inp->setDebugName(ReplaceString(name, ".", "_"));
    }
  }
  for (const auto& inp : graph->outputs()) {
    auto name = inp->debugName();
    if (name.find('.') != std::string::npos)
      inp->setDebugName(ReplaceString(name, ".", "_"));
  }

  torch::jit::Inline(*graph);
  torch::jit::RemoveTensorMutation(graph);
  return std::make_pair(graph, module);
}

void make_small_allocs(int num_loops, int thread_id) {
  auto oversize_threshold = 1097152;
  for (int i = 0; i < num_loops; ++i) {
    auto oversize_input = at::empty({oversize_threshold - 1}, at::kCPU);
    auto oversize_input1 = at::empty({oversize_threshold - 1}, at::kCPU);
    auto oversize_input2 = at::empty({oversize_threshold - 1}, at::kCPU);
    //    auto res1 = 2 * oversize_input;
    //    auto res2 = 2 * oversize_input1;
    //    auto res3 = 2 * oversize_input2;
  }
}

inline Stack createStack(std::vector<at::Tensor>&& list) {
  return Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

inline std::vector<IValue> stack(int batch_size, int hw) {
  auto input = at::randn({batch_size, 3, hw, hw}, at::kCPU);
  auto stack = createStack({input});
  return stack;
}

void run_model(std::string model_name, int num_loops, int thread_id, int batch_size, int hw) {
  // run once to type info
  torch::NoGradGuard no_grad;
  auto g =
      get_graph_from_model(
          "/home/mlevental/dev_projects/pytorch_memory_planning/models/" +
          model_name + ".pt")
          .first;

  Code cd(g, model_name);
  InterpreterState is{cd};
  for (int i = 0; i < num_loops; ++i) {
    auto inp_stack = stack(batch_size, hw);
    is.run(inp_stack);
    is.reset_frame(cd);
  }
}

int str2int(const std::string& str) {
  std::stringstream ss(str);
  int num;
  if ((ss >> num).fail()) {
    CAFFE_THROW("couldn't parse num");
  }
  return num;
}

std::string int2str(int num) {
  std::stringstream ss;
  if ((ss << num).fail()) {
    CAFFE_THROW("couldn't parse num");
  }
  return ss.str();
}

int main(int argc, const char* argv[]) {
  std::string name{argv[1]};
  auto num_workers = str2int(argv[2]);
  auto num_loops = str2int(argv[3]);
  auto batch_size = str2int(argv[4]);
  auto hw = str2int(argv[5]);
  auto thresh = str2int(argv[6]);

  std::vector<std::thread> threads;
  threads.reserve(num_workers);
  for (int thread_id = 0; thread_id < num_workers; ++thread_id) {
    threads.emplace_back(run_model, name, num_loops, thread_id, batch_size, hw);
//        threads.emplace_back(make_small_allocs, num_loops, thread_id);
  }
  if (!threads.empty()) {
    for (int thread_id = 0; thread_id < num_workers; ++thread_id) {
      threads[thread_id].join();
    }
  }
  std::string normal;
  if (thresh < 0) {
    normal = "normal";
  } else {
    normal = "one_arena";
  }
  je_malloc_print_stats(
      "/home/mlevental/dev_projects/pytorch_memory_planning/je_malloc_runs/"+ name + "/" + normal + "_" + int2str(num_workers) + "_" + int2str(num_loops) + "_" + int2str(batch_size) + "_" + int2str(hw));
}

//const size_t num_total_iters = iters * pt_inputs.size() * num_threads;
//const double ms_per_iter = ptms / num_total_iters;
//msPerIter.push_back(ms_per_iter);
//
//LOG(INFO) << "PyTorch run finished. Milliseconds per iter: " << ms_per_iter
//          << ". Iters per second: " << (1000.0 / ms_per_iter);
//if (generate_ai_pep_output) {
//  auto json_string =
//      jsonOutputPEP("NET", "latency", "millisecond per iter", ms_per_iter);
//  LOG(INFO) << "PyTorchObserver " << json_string;
//}
//}
//if (repetitions > 1) {
//  folly::StreamingStats<double> msPerIterStats(
//      msPerIter.begin(), msPerIter.end());
//  LOG(INFO) << "Mean milliseconds per iter: " << msPerIterStats.mean()
//            << ", standard deviation: "
//            << msPerIterStats.sampleStandardDeviation();
//}