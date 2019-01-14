#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <re2/re2.h>

#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/fb/predictor/Transforms.h"
#include "caffe2/fb/predictor/multithread_bench/aibench_output_formatter.h"
#include "caffe2/fb/predictor/multithread_bench/net_loader.h"
#include "caffe2/fb/predictor/multithread_bench/perf_profiler.h"
#include "caffe2/fb/predictor/multithread_bench/predictor_pair_emulator.h"
#include "caffe2/fb/predictor/multithread_bench/prod_net_supplier.h"
#include "caffe2/fb/predictor/multithread_bench/stack_analyzer.h"
#include "caffe2/predictor/emulator/benchmark.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

C10_DECLARE_int(caffe2_logging_network_dyno_sampling_rate);
C10_DECLARE_int(caffe2_logging_operator_dyno_sampling_rate);
C10_DECLARE_bool(caffe2_predictor_cleanup_activations);

C10_DEFINE_int(num_models, 300, "Number of models to emulate");

C10_DEFINE_string(
    production_nets_path,
    "/tmp/caffe2_production_nets",
    "Path to where we store the caffe2 production nets");
C10_DEFINE_bool(
    update_prod,
    false,
    "Force to download the latest nets from production.");
C10_DEFINE_bool(use_prod, true, "Use production nets.");
C10_DEFINE_string(
    prod_tw_job,
    "adindexer",
    "A string to filter the production tupperware jobs to select from. "
    "A model will be matched if its tupperware job contains the given string");
C10_DEFINE_bool(
    dump_nets,
    false,
    "If set, dump net proto before and after onnxifi transformation");

C10_DEFINE_bool(
    json_output,
    false,
    "Print json output so that aibench can log the results to scuba.");

// For async scheduling test
C10_DEFINE_string(
    benchmark_net_type,
    "",
    "If used, we will use async_scheduling instead simple net for predict net");

C10_DEFINE_bool(
    benchmark_async_deferrable_mode,
    true,
    "If used, use deferrable_mode for DFS scheduling in async_scheduling net");

C10_DEFINE_int(
    benchmark_async_workers,
    -1,
    "Set number of worker threads for the thread pool in asyn_scheduling net");

C10_DEFINE_string(
    benchmark_model_transformation,
    "fp16", // or "int8"
    "Transformation for predictors");

C10_DEFINE_bool(
    rowwise_quantization,
    false,
    "Set if we use rowwise quantizationf for fc operators");

C10_DEFINE_string(
    dnnlowp_weight_quantization_kind,
    "min_max",
    "Set the quantization scheme for weights");

C10_DEFINE_string(
    dnnlowp_activation_quantization_kind,
    "min_max",
    "Set the quantization scheme for activations");

C10_DEFINE_int(dnnlowp_weight_precision, 8, "Set the precision for weights");

C10_DEFINE_int(
    dnnlowp_activation_precision,
    8,
    "Set the precision for activations");

// For debug use only
C10_DEFINE_string(
    profiling_level,
    "TIMER",
    "'TIMER', 'PERF', 'PERF_DETAILED_OVERHEAD' or 'PERF_DETAILED_ALL'");
enum class ProfilingLevel {
  TIMER,
  PERF,
  PERF_DETAILED_OVERHEAD,
  PERF_DETAILED_ALL
};
ProfilingLevel profiling_level() {
  static const std::unordered_map<std::string, ProfilingLevel> values{
      {"TIMER", ProfilingLevel::TIMER},
      {"PERF", ProfilingLevel::PERF},
      {"PERF_DETAILED_OVERHEAD", ProfilingLevel::PERF_DETAILED_OVERHEAD},
      {"PERF_DETAILED_ALL", ProfilingLevel::PERF_DETAILED_ALL}};
  static const ProfilingLevel value = values.at(FLAGS_profiling_level);
  return value;
}
const std::string perf_data_path = "/tmp/perf.data";
C10_DEFINE_string(perf_unit, "cpu-clock", "'cpu-clock' or 'cycles'");

namespace caffe2 {
namespace emulator {

class BlackBoxPredictorBenchmarkRunner : public BenchmarkRunner {
 protected:
  void pre_benchmark_setup() override {
    if (!FLAGS_caffe2_predictor_cleanup_activations) {
      // we don't expect to have any allocations happening in the case
      // predictor doesn't cleanup activations after each run
      // Setting this flag allows us to see if this is not a case
      // Allocations have been found to work slow with NUMA, this is why
      // we try to pay a lot of attention to them
      FLAGS_caffe2_report_cpu_memory_usage = true;
    }
  }

  void post_benchmark_cleanup() override {
    // No need to report it any more as allocations are fine outside of the
    // main part of the benchmark
    FLAGS_caffe2_report_cpu_memory_usage = false;
  }
};

void run() {
  BenchmarkParam param;
  std::unique_ptr<StackAnalyzer> analyzer = nullptr;

  std::string script_patch = perf_data_path + ".script";
  switch (profiling_level()) {
    case ProfilingLevel::TIMER:
      param.profiler = std::unique_ptr<Profiler>(new TimeProfiler());
      break;
    case ProfilingLevel::PERF_DETAILED_ALL:
    case ProfilingLevel::PERF_DETAILED_OVERHEAD:
      if (FLAGS_perf_unit != "cycles" && FLAGS_perf_unit != "cpu-clock") {
        throw std::invalid_argument(
            "perf_unit argument must be 'cycles' or 'cpu-clock'.");
      }
      analyzer = make_unique<StackAnalyzer>(
          script_patch,
          profiling_level() == ProfilingLevel::PERF_DETAILED_OVERHEAD,
          "caffe2::BlackBoxPredictor::operator()",
          FLAGS_perf_unit);
      // no need to break; continue to check path and create profiler
    case ProfilingLevel::PERF:
      if (!check_path_valid(perf_data_path)) {
        throw std::invalid_argument("invalid path " + perf_data_path);
      }
      if (!check_path_valid(script_patch)) {
        throw std::invalid_argument("invalid path " + script_patch);
      }
      param.profiler = std::unique_ptr<Profiler>(
          new PerfProfiler(perf_data_path, script_patch));
      break;
    default:
      throw std::logic_error(
          "invalid profiling level " +
          caffe2::to_string(static_cast<int>(profiling_level())));
  }

  // Build supplier
  std::unique_ptr<NetSupplier> supplier = nullptr;
  // Using async scheduling net if specified
  auto async_net_mutator = [](NetDef* net) {
    if (!FLAGS_benchmark_net_type.empty()) {
      net->set_type(FLAGS_benchmark_net_type);
      if (FLAGS_benchmark_async_workers > 0) {
        net->set_num_workers(FLAGS_benchmark_async_workers);
      }
      if (FLAGS_benchmark_async_deferrable_mode) {
        auto* arg = net->add_arg();
        arg->set_name("deferrable_mode");
        arg->set_i(1);
      }
    }
  };
  if (!FLAGS_use_prod) {
    CAFFE_ENFORCE(
        !FLAGS_update_prod,
        "cannot update production nets if they are not used");
    // Create the data filler
    caffe2::NetDef run_net;
    CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_run_net, &run_net));
    unique_ptr<Filler> filler;
    if (!FLAGS_input_dims.empty() || !FLAGS_input_types.empty()) {
      if (!check_path_valid(FLAGS_input_dims, false)) {
        throw std::invalid_argument(
            "cannot find input_dims file " + FLAGS_input_dims);
      }
      if (!check_path_valid(FLAGS_input_types, false)) {
        throw std::invalid_argument(
            "cannot find input_types file " + FLAGS_input_types);
      }
      std::ifstream dims_in(FLAGS_input_dims);
      std::string dims(
          (std::istreambuf_iterator<char>(dims_in)),
          std::istreambuf_iterator<char>());
      std::ifstream types_in(FLAGS_input_types);
      std::string types(
          (std::istreambuf_iterator<char>(types_in)),
          std::istreambuf_iterator<char>());
      filler = std::unique_ptr<Filler>(new DataRandomFiller(
          run_net, parse_json_dims(dims), parse_json_types(types)));
    } else {
      // use data_net and init_net
      NetDef init_net;
      NetDef data_net;
      CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
      CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_data_net, &data_net));
      filler = std::unique_ptr<Filler>(
          new DataNetFiller(std::move(init_net), std::move(data_net)));
    }

    supplier = caffe2::make_unique<MutatingNetSupplier>(
        caffe2::make_unique<SingleNetSupplier>(std::move(filler), run_net),
        async_net_mutator);
  } else {
    CAFFE_ENFORCE(!FLAGS_prod_tw_job.empty(), "needs to specify the tw jobs");
    std::string nets_dir;
    boost::filesystem::path dir_path(FLAGS_production_nets_path);
    if (FLAGS_update_prod || !is_directory(dir_path)) {
      NetLoader loader(FLAGS_production_nets_path);
      nets_dir = loader.load_prod_nets_and_save(FLAGS_prod_tw_job);
    } else {
      for (auto& entry : boost::make_iterator_range(
               boost::filesystem::directory_iterator(dir_path), {})) {
        std::ostringstream stream;
        stream << entry;
        auto path = replace(stream.str(), "\"", "");
        if (path > nets_dir &&
            re2::RE2::FullMatch(
                path,
                std::string(".*[0-9]{4}-[0-9]{2}-[0-9]{2}") + FOLDER_DELIMITER +
                    FLAGS_prod_tw_job + "$")) {
          nets_dir = path;
        }
      }
      if (nets_dir.empty()) {
        NetLoader loader(FLAGS_production_nets_path);
        nets_dir = loader.load_prod_nets_and_save(FLAGS_prod_tw_job);
      }
    }
    supplier = caffe2::make_unique<MutatingNetSupplier>(
        caffe2::make_unique<ProductionNetSupplier>(nets_dir),
        async_net_mutator);
  }
  auto transformer = [](NetDef* net,
                        Workspace* ws,
                        std::vector<std::string>* input_names,
                        std::vector<std::string>* output_names,
                        const std::vector<TensorCPU>& input_tensors,
                        const std::unordered_set<int>& blacklist) {
    NetDef net_org = *net;

    // Transformation
    if (FLAGS_dump_nets) {
      WriteProtoToTextFile(*net, "before.pb_txt");
    }
    if (FLAGS_benchmark_model_transformation == "fp16" ||
        FLAGS_benchmark_model_transformation == "int8") {
      for (int i = 0; i < net_org.op_size(); ++i) {
        if (net_org.op(i).type() == "FCTransposed") {
          auto* op = net_org.mutable_op(i);
          string transposed_weight_blob_name = op->input(1);
          auto* transposed_weight_blob =
              ws->GetBlob(transposed_weight_blob_name);
          const auto& transposed_weight_tensor =
              transposed_weight_blob->Get<TensorCPU>();
          auto* in_data = transposed_weight_tensor.data<float>();
          string weight_blob_name = transposed_weight_blob_name + "_org";
          if (transposed_weight_blob_name.find("_transposed") != string::npos) {
            weight_blob_name = transposed_weight_blob_name.substr(
                0, transposed_weight_blob_name.length() - 11);
          }
          auto* weight_blob = ws->CreateBlob(weight_blob_name);

          auto N = transposed_weight_tensor.sizes()[0];
          auto M = transposed_weight_tensor.sizes()[1];

          auto* weight_tensor = BlobGetMutableTensor(weight_blob, CPU);
          weight_tensor->Resize(M, N);

          auto* out_data = weight_tensor->mutable_data<float>();

          for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
              out_data[j * N + i] = in_data[i * M + j];
            }
          }
          op->set_type("FC");
          *op->mutable_input(1) = weight_blob_name;
        }
      }
      if (FLAGS_benchmark_model_transformation == "fp16") {
        vector<std::shared_ptr<NetDef>> all_nets;
        all_nets.push_back(std::make_shared<NetDef>(net_org));
        auto mapping = FCTransformed("FbFCPacked", all_nets, ws);
        for (int i = 0; i < net_org.op_size(); ++i) {
          if (net_org.op(i).type() == "FC") {
            string weight_blob_name = net_org.op(i).input(1);
            if (mapping.count(weight_blob_name)) {
              LOG(INFO) << "fp16 transformation of weight " << weight_blob_name
                        << " to " << mapping[weight_blob_name];
              auto* op = net->mutable_op(i);
              op->set_type("FbFCPacked");
              *op->mutable_input(1) = mapping[weight_blob_name];
            }
          }
        }
      } else {
        // dynamica quantization
        for (int i = 0; i < net_org.op_size(); ++i) {
          if (net_org.op(i).type() == "FC") {
            auto* op = net->mutable_op(i);
            if (FLAGS_rowwise_quantization) {
              op->set_engine("DNNLOWP_ROWWISE");
            } else {
              op->set_engine("DNNLOWP");
            }
            auto* dequantize_output_arg = op->add_arg();
            dequantize_output_arg->set_name("dequantize_output");
            dequantize_output_arg->set_i(1);
            auto* error_arg = op->add_arg();
            error_arg->set_name("measure_quantization_error");
            error_arg->set_i(1);
            if (i < net_org.op_size() && net_org.op(i + 1).type() == "Relu") {
              auto* followed_by_relu_arg = op->add_arg();
              followed_by_relu_arg->set_name("followed_by");
              followed_by_relu_arg->set_s("Relu");
            }
            auto* weight_precision_arg = op->add_arg();
            weight_precision_arg->set_name("weight_precision");
            weight_precision_arg->set_i(FLAGS_dnnlowp_weight_precision);
            auto* activation_precision_arg = op->add_arg();
            activation_precision_arg->set_name("activation_precision");
            activation_precision_arg->set_i(FLAGS_dnnlowp_activation_precision);
            auto* weight_kind_arg = op->add_arg();
            weight_kind_arg->set_name("weight_quantization_kind");
            weight_kind_arg->set_s(FLAGS_dnnlowp_weight_quantization_kind);
            auto* activation_kind_arg = op->add_arg();
            activation_kind_arg->set_name("activation_quantization_kind");
            activation_kind_arg->set_s(
                FLAGS_dnnlowp_activation_quantization_kind);
          }
        }
      }
    } else {
      LOG(WARNING) << "Only fp16 and int8 transformations are supported now.";
    }
    if (FLAGS_dump_nets) {
      WriteProtoToTextFile(*net, "after.pb_txt");
    }
  };
  param.emulator = std::unique_ptr<Emulator>(new PredictorPairEmulator(
      std::move(supplier),
      transformer,
      FLAGS_threads,
      FLAGS_num_models,
      FLAGS_num_loading_threads));

  // build OutputFormatter
  if (FLAGS_json_output) {
    param.formatter =
        std::unique_ptr<OutputFormatter>(new AIBenchOutputFormatter());
  } else {
    param.formatter =
        std::unique_ptr<OutputFormatter>(new StdOutputFormatter());
  }

  // Run benchmark
  BlackBoxPredictorBenchmarkRunner runner;
  runner.benchmark(param);

  // Extra analysis if enabled; debug only
  if (analyzer.get() != nullptr) {
    if (!check_path_valid(script_patch, false)) {
      throw std::invalid_argument("cannot find perf script " + script_patch);
    }
    const auto sorted_weights = analyzer->analyze();

    LOG(INFO) << "\n\n==============\nPerf Results\n==============";
    const size_t limit = 20;
    size_t printed = 0;
    for (const auto& weight : sorted_weights) {
      LOG(INFO) << weight.second << "%\t\t" << weight.first;
      if (++printed >= limit) {
        break;
      }
    }
  }
}

} // namespace emulator
} // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  FLAGS_caffe2_logging_network_dyno_sampling_rate = 0;
  FLAGS_caffe2_logging_operator_dyno_sampling_rate = 0;

  // TODO: a hack to initialized flags so that production benchmark won't fail
  const std::string tmp_dir = "/tmp/predictor_benchmark_example/";
  if (FLAGS_run_net == "" && FLAGS_init_net == "" && FLAGS_data_net == "") {
    FLAGS_run_net = tmp_dir + "run_net.pb";
    FLAGS_init_net = tmp_dir + "init_net.pb";
    FLAGS_data_net = tmp_dir + "data_init.pb";
  }
  if (FLAGS_numa_different) {
    FLAGS_caffe2_cpu_numa_enabled = true;
    c10::NUMABind(FLAGS_numa_default_node);
  }
  caffe2::emulator::run();
  return 0;
}
