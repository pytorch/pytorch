#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <assert.h>
#include <torch/deploy.h>

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

#include <torch/script.h>

typedef void (*function_type)(const char*);

bool cuda = false;

constexpr auto latency_p = {
    25.,
    50.,
    95.}; //{1., 5., 25., 50., 75., 90., 95., 99., 99.25, 99.5, 99.75, 99.9};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct Report {
  std::string benchmark;
  std::string strategy;
  size_t n_threads;
  size_t items_completed;
  double work_items_per_second;
  std::vector<double> latencies;
  static void report_header(std::ostream& out) {
    out << "benchmark, strategy, n_threads, work_items_completed, work_items_per_second";
    for (double l : latency_p) {
      out << ", p" << l << "_latency";
    }
    out << ", device\n";
  }
  void report(std::ostream& out) {
    out << benchmark << ", " << strategy << ", " << n_threads << ", "
        << items_completed << ", " << work_items_per_second;
    for (double l : latencies) {
      out << ", " << l;
    }
    out << ", " << (cuda ? "cuda" : "cpu") << "\n";
  }
};

const int min_items_to_complete = 1;

struct RunPython {
  static torch::deploy::ReplicatedObj load_and_wrap(
      torch::deploy::Package& package) {
    auto I = package.acquire_session();
    auto obj = I.self.attr("load_pickle")({"model", "model.pkl"});
    if (cuda) {
      obj = I.global("gpu_wrapper", "GPUWrapper")({obj});
    }
    return I.create_movable(obj);
  }
  RunPython(
      torch::deploy::Package& package,
      std::vector<at::IValue> eg,
      const torch::deploy::Interpreter* interps)
      : obj_(load_and_wrap(package)), eg_(std::move(eg)), interps_(interps) {}
  void operator()(int i) {
    auto I = obj_.acquire_session();
    if (cuda) {
      std::vector<at::IValue> eg2 = {i};
      eg2.insert(eg2.end(), eg_.begin(), eg_.end());
      I.self(eg2);
    } else {
      I.self(eg_);
    }
  }
  torch::deploy::ReplicatedObj obj_;
  std::vector<at::IValue> eg_;
  const torch::deploy::Interpreter* interps_;
};

// def to_device(i, d):
//     if isinstance(i, torch.Tensor):
//         return i.to(device=d)
//     elif isinstance(i, (tuple, list)):
//         return tuple(to_device(e, d) for e in i)
//     else:
//         raise RuntimeError('inputs are weird')

static torch::IValue to_device(const torch::IValue& v, torch::Device to);

static std::vector<torch::IValue> to_device_vec(
    at::ArrayRef<torch::IValue> vs,
    torch::Device to) {
  std::vector<torch::IValue> results;
  for (const torch::IValue& v : vs) {
    results.push_back(to_device(v, to));
  }
  return results;
}

static torch::IValue to_device(const torch::IValue& v, torch::Device to) {
  if (v.isTensor()) {
    return v.toTensor().to(to);
  } else if (v.isTuple()) {
    auto tup = v.toTuple();
    return c10::ivalue::Tuple::create(to_device_vec(tup->elements(), to));
  } else if (v.isList()) {
    auto converted = to_device_vec(v.toListRef(), to);
    torch::List<torch::IValue> result(v.toList().elementType());
    for (const torch::IValue& v : converted) {
      result.push_back(v);
    }
    return result;
  } else {
    TORCH_INTERNAL_ASSERT(false, "cannot to_device");
  }
}

static bool exists(const std::string& fname) {
  std::fstream jit_file(fname);
  return jit_file.good();
}

struct RunJIT {
  RunJIT(const std::string& file_to_run, std::vector<torch::IValue> eg)
      : eg_(std::move(eg)) {
    if (!cuda) {
      models_.push_back(torch::jit::load(file_to_run + "_jit"));
    } else {
      for (int i = 0; i < 2; ++i) {
        auto d = torch::Device(torch::DeviceType::CUDA, i);
        std::stringstream qualified;
        qualified << file_to_run << "_jit_" << i;
        auto loaded = exists(qualified.str())
            ? torch::jit::load(qualified.str(), d)
            : torch::jit::load(file_to_run + "_jit", d);
        loaded.to(d);
        models_.push_back(loaded);
      }
    }
  }
  void operator()(int i) {
    if (cuda) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      int device_id = i % models_.size();
      auto d = torch::Device(torch::DeviceType::CUDA, device_id);
      to_device(
          models_[device_id].forward(to_device_vec(eg_, d)),
          torch::DeviceType::CPU);
    } else {
      models_[0].forward(eg_);
    }
  }
  std::vector<at::IValue> eg_;
  std::vector<torch::jit::Module> models_;
};

struct Benchmark {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Benchmark(
      torch::deploy::InterpreterManager& manager,
      size_t n_threads,
      std::string strategy,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::string file_to_run,
      size_t n_seconds = 5)
      : manager_(manager),
        n_threads_(n_threads),
        strategy_(strategy),
        file_to_run_(file_to_run),
        n_seconds_(n_seconds),
        should_run_(true),
        items_completed_(0),
        reached_min_items_completed_(0) {
    if (strategy == "one_python") {
      manager.debugLimitInterpreters(1);
    } else if (strategy == "multi_python") {
      manager.debugLimitInterpreters(n_threads_);
    }
  }

  Report run() {
    pthread_barrier_init(&first_run_, nullptr, n_threads_ + 1);

    torch::deploy::Package package = manager_.load_package(file_to_run_);

    std::vector<at::IValue> eg;
    {
      auto I = package.acquire_session();

      eg = I.global("builtins", "tuple")(
                I.self.attr("load_pickle")({"model", "example.pkl"}))
               .toIValue()
               .toTuple()
               ->elements();
    }

    if (strategy_ == "jit") {
      run_one_work_item = RunJIT(file_to_run_, std::move(eg));
    } else {
      run_one_work_item =
          RunPython(package, std::move(eg), manager_.all_instances().data());
    }

    std::vector<std::vector<double>> latencies(n_threads_);

    for (size_t i = 0; i < n_threads_; ++i) {
      threads_.emplace_back([this, &latencies, i] {
        torch::NoGradGuard guard;
        // do initial work
        run_one_work_item(i);

        pthread_barrier_wait(&first_run_);
        size_t local_items_completed = 0;
        while (should_run_) {
          auto begin = std::chrono::steady_clock::now();
          run_one_work_item(i);
          auto end = std::chrono::steady_clock::now();
          double work_seconds =
              std::chrono::duration<double>(end - begin).count();
          latencies[i].push_back(work_seconds);
          local_items_completed++;
          if (local_items_completed == min_items_to_complete) {
            reached_min_items_completed_++;
          }
        }
        items_completed_ += local_items_completed;
      });
    }

    pthread_barrier_wait(&first_run_);
    auto begin = std::chrono::steady_clock::now();
    auto try_stop_at = begin + std::chrono::seconds(n_seconds_);
    std::this_thread::sleep_until(try_stop_at);
    for (int i = 0; reached_min_items_completed_ < n_threads_; ++i) {
      std::this_thread::sleep_until(
          begin + (i + 2) * std::chrono::seconds(n_seconds_));
    }
    should_run_ = false;
    for (std::thread& thread : threads_) {
      thread.join();
    }
    auto end = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end - begin).count();
    Report report;
    report.benchmark = file_to_run_;
    report.strategy = strategy_;
    report.n_threads = n_threads_;
    report.items_completed = items_completed_;
    report.work_items_per_second = items_completed_ / total_seconds;
    reportLatencies(report.latencies, latencies);
    run_one_work_item = nullptr;
    return report;
  }

 private:
  void reportLatencies(
      std::vector<double>& results,
      const std::vector<std::vector<double>>& latencies) {
    std::vector<double> flat_latencies;
    for (const auto& elem : latencies) {
      flat_latencies.insert(flat_latencies.end(), elem.begin(), elem.end());
    }
    std::sort(flat_latencies.begin(), flat_latencies.end());
    for (double target : latency_p) {
      size_t idx = size_t(flat_latencies.size() * target / 100.0);
      double time = flat_latencies.size() == 0
          ? 0
          : flat_latencies.at(std::min(flat_latencies.size() - 1, idx));
      results.push_back(time);
    }
  }
  torch::deploy::InterpreterManager& manager_;
  size_t n_threads_;
  std::string strategy_;
  std::string file_to_run_;
  size_t n_seconds_;
  pthread_barrier_t first_run_;
  std::atomic<bool> should_run_;
  std::atomic<size_t> items_completed_;
  std::atomic<size_t> reached_min_items_completed_;
  std::vector<std::thread> threads_;
  std::function<void(int)> run_one_work_item;
};

int main(int argc, char* argv[]) {
  int max_thread = atoi(argv[1]);
  cuda = std::string(argv[2]) == "cuda";
  bool jit_enable = std::string(argv[3]) == "jit";
  Report::report_header(std::cout);
  torch::deploy::InterpreterManager manager(max_thread);

  // make sure gpu_wrapper.py is in the import path
  for (auto& interp : manager.all_instances()) {
    auto I = interp.acquire_session();
    I.global("sys", "path").attr("append")({"torch/csrc/deploy/example"});
  }

  auto n_threads = {1, 2, 4, 8, 16, 32, 40};
  for (int i = 4; i < argc; ++i) {
    std::string model_file = argv[i];
    for (int n_thread : n_threads) {
      if (n_thread > max_thread) {
        continue;
      }
      for (std::string strategy : {"one_python", "multi_python", "jit"}) {
        if (strategy == "jit") {
          if (!jit_enable) {
            continue;
          }
          if (!exists(model_file + "_jit")) {
            continue;
          }
        }
        Benchmark b(manager, n_thread, strategy, model_file);
        Report r = b.run();
        r.report(std::cout);
      }
    }
  }
  return 0;
}
