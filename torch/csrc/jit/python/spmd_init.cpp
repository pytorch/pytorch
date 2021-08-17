#include <torch/csrc/jit/python/spmd_init.h>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/ThreadLocalState.h>

#include <ATen/record_function.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <cuda.h>

#include <sstream>


// STOPPED HERE TUESDAY NIGHT. Outstanding issues:
// * Some sort of exception corruption when running twice?
// * Gigantic cudaFree calls that seem to be mutually exclusive ?? Coming from linear. KP: "cudaFree is death" - Natalia

namespace torch {
namespace jit {

using Stack = std::vector<c10::IValue>;

// Heavily inspired by createStackForSchema
Stack createStackFromIValueArgsKwargs(
        const FunctionSchema& schema,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs) {
    size_t all_arguments = args.size() + kwargs.size();
    if (all_arguments > schema.arguments().size()) {
        throw schema_match_error(c10::str(
            schema.name(),
            "() expected at most ",
            schema.arguments().size(),
            " argument(s) but received ",
            all_arguments,
            " argument(s). Declaration: ",
            schema));
    }

    Stack stack;
    stack.reserve(schema.arguments().size());

    int64_t arg_idx = 0;
    for (const auto& arg : args) {
        // ...but refuse to do it if the schema says that this was supposed
        // to be keyword only
        if (schema.arguments()[arg_idx].kwarg_only()) {
        throw schema_match_error(c10::str(
            schema.name(),
            "() takes ",
            arg_idx,
            " positional argument(s) but ",
            args.size(),
            " was/were given.  Declaration: ",
            schema));
        }
        // Use the type information from the schema to convert the PyObject.
        push(stack, arg);
        arg_idx++;
    }

    // Now for every remaining non-positional argument in the schema, look for it
    // in the kwargs dict and push it if found, or use its default value if it
    // has one.
    size_t consumed_kwargs = 0;
    for (size_t i = stack.size(); i < schema.arguments().size(); ++i) {
        const auto& arg = schema.arguments()[i];
        if (kwargs.count(arg.name().c_str())) {
            push(stack, kwargs.at(arg.name().c_str()));
            consumed_kwargs += 1;
        } else if (arg.default_value()) {
            push(stack, *arg.default_value());
        } else {
            throw schema_match_error(c10::str(
                schema.name(),
                "() is missing value for argument '",
                arg.name(),
                "'. Declaration: ",
                schema));
        }
    }

    if (consumed_kwargs != kwargs.size()) {
        std::vector<std::string> names;
        for (const auto& kwarg : kwargs) {
        names.emplace_back(kwarg.first);
        }
        throw schema_match_error(schema.findErrorInKwargs(names));
    }

    return stack;
}

c10::IValue ivalue_replace_strings(const c10::IValue value, const std::function<IValue (const std::string&)>& visitor) {
    // TODO: recursive descent
    // Not even sure what's going on here, toTypeInferredIValue might just
    // fail on heterogeneously typed inputs.
    if (value.isString()) {
        return visitor(value.toStringRef());
    } else {
        return value;
    }
}

struct CallsiteDescr {
  public:
    // COMMAND
    CallsiteDescr(
            std::string target, std::string overload, py::tuple args, py::dict kwargs, std::string output_id)
        : kind(COMMAND),
          target_(std::move(target)),
          overload_(std::move(overload)),
          output_id(std::move(output_id))
    {
        for (auto obj : args) {
            args_.emplace_back(toTypeInferredIValue(std::move(obj)));
        }

        for (auto &entry : kwargs) {
            kwargs_[py::cast<std::string>(entry.first)] = toTypeInferredIValue(std::move(entry.second));
        }
    }

    // TERMINATE
    CallsiteDescr(bool terminate) : kind(TERMINATE) {}

    // LOAD_VALUE
    CallsiteDescr(std::string identifier, c10::IValue value)
        : kind(LOAD_VALUE),
          args_({std::move(value)}),
          output_id(std::move(identifier)) {}

    // DEL_VALUE
    CallsiteDescr(std::string identifier)
        : kind(DEL_VALUE),
          output_id(std::move(identifier)) {}

    // Get return value
    CallsiteDescr(int /*dummy*/, std::string identifier)
        : kind(RETURN_VALUE),
          output_id(std::move(identifier)) {}

    c10::IValue call(std::function<c10::IValue(const std::string&)> arg_remap_fn) {
        std::vector<c10::IValue> remapped_args;
        for (auto& arg : args_) {
            remapped_args.emplace_back(ivalue_replace_strings(arg, arg_remap_fn));
        }
        std::unordered_map<std::string, c10::IValue> remapped_kwargs;
        for (auto & pair : kwargs_) {
            remapped_kwargs[pair.first] = ivalue_replace_strings(pair.second, arg_remap_fn);
        }

        // std::stringstream ss;
        // ss << schema_str_ << "\nArgs: ";
        // for (auto &arg : remapped_args) {
        //     ss << arg.type()->annotation_str() << ", ";
        // }
        // ss << " kwargs: ";
        // for (auto &kwarg_pair : remapped_kwargs) {
        //     ss << kwarg_pair.first << "=" << kwarg_pair.second.type()->annotation_str() << ", ";
        // }
        // ss << " stack: ";
        // for (auto &stack_item : stack) {
        //     ss << stack_item.type()->annotation_str() << ", ";
        // }
        // ss << "\n";
        // std::cout << ss.str() << std::flush;

        auto op = findOperatorFor({target_, overload_});
        auto operation = op->getOperation();
        auto stack = createStackFromIValueArgsKwargs(op->schema(), remapped_args, remapped_kwargs);
        operation(&stack);

        // ss = std::stringstream();
        // ss << "end stack: ";
        // for (auto &stack_item : stack) {
        //     ss << stack_item.type()->annotation_str() << ", ";
        // }
        // ss << "\n\n";
        // std::cout << ss.str() << std::flush;

        if (stack.size() != 1) {
            throw std::runtime_error("Expected single output");
        }

        return std::move(stack[0]);
    }

    enum {
        TERMINATE,
        COMMAND,
        LOAD_VALUE,
        DEL_VALUE,
        RETURN_VALUE
    } kind;

    std::string target_;
    std::string overload_;

    std::vector<c10::IValue> args_;
    std::unordered_map<std::string, c10::IValue> kwargs_;

    std::string output_id;

    at::ThreadLocalState tls_;
};

class ThreadedCUDAInstance {
  public:
    ThreadedCUDAInstance(int64_t device_id)
        : device_id(device_id) {

        auto worker_fn = [this]() {
            cudaSetDevice(this->device_id);

            c10::optional<std::string> rv_str;
            while (true) {
                std::vector<CallsiteDescr> command_queue;
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [this]{ return !queue.empty(); });

                    while (!queue.empty()) {
                        auto &call = queue.front();
                        command_queue.emplace_back(std::move(call));
                        queue.pop();
                    }
                }

                for (auto& call : command_queue) {
                    // TODO this is really dumb
                    at::ThreadLocalStateGuard tls_guard(call.tls_);
                    auto start = std::chrono::high_resolution_clock::now();

                    std::stringstream ss;
                    ss << this->device_id << " ";
                    switch (call.kind) {
                        case CallsiteDescr::TERMINATE: ss << "TERMINATE"; break;
                        case CallsiteDescr::COMMAND: ss << "COMMAND"; break;
                        case CallsiteDescr::LOAD_VALUE: ss << "LOAD_VALUE"; break;
                        case CallsiteDescr::DEL_VALUE: ss << "DEL_VALUE"; break;
                        case CallsiteDescr::RETURN_VALUE: ss << "RETURN_VALUE"; break;
                    }

                    if (call.kind == CallsiteDescr::COMMAND) {
                        ss << " " << call.target_;
                    }

                    if (!call.overload_.empty()) {
                        ss << "." << call.overload_;
                    }

                    ss << " " << call.output_id;

                    auto descr_str = ss.str();

                    RECORD_FUNCTION(descr_str, {});
                    // std::cout << descr_str << std::flush;

                    if (call.kind == CallsiteDescr::TERMINATE) {
                        return;
                    } else if (call.kind == CallsiteDescr::COMMAND) {
                        auto remap_fn = [this](const std::string& name) -> c10::IValue {
                            return env.at(name);
                        };

                        env[call.output_id] = call.call(remap_fn);
                    } else if (call.kind == CallsiteDescr::LOAD_VALUE) {
                        env[call.output_id] = std::move(call.args_.at(0));
                    } else if (call.kind == CallsiteDescr::DEL_VALUE) {
                        env.erase(call.output_id);
                    } else if (call.kind == CallsiteDescr::RETURN_VALUE) {
                        if (env.count(call.output_id)) {
                            std::unique_lock<std::mutex> lock(value_m);
                            return_value = env.at(call.output_id);
                            value_cv.notify_one();
                        } else {
                            rv_str = call.output_id;
                        }
                    }

                    // ss = std::stringstream();
                    // ss << this->device_id << " end\n";
                    // std::cout << ss.str() << std::flush;

                    if (rv_str && env.count(*rv_str)) {
                        {
                            std::unique_lock<std::mutex> lock(value_m);
                            return_value = env.at(*rv_str);
                            rv_str = c10::nullopt;
                        }
                        value_cv.notify_one();
                    }

                    // auto end = std::chrono::high_resolution_clock::now();
                    // ss = std::stringstream();
                    // ss << std::chrono::duration<double>(end-start).count() * 1e6 << " " << descr_str << "\n";
                    // std::cout << ss.str() << std::flush;
                    // if (env.count(call.output_id) and env[call.output_id].isTensor()) {
                    //     std::stringstream ss2;
                    //     ss2 << this->device_id << " " << env[call.output_id].toTensor().device().str() << " " << call.output_id << "\n";
                    //     std::cout << ss2.str() << std::flush;
                    // }
                }
            }
        };

        std::stringstream ss;
        ss << "cuda:" << device_id;
        device_str_ = ss.str();

        worker_thread_ = std::thread(worker_fn);
    }

    void execCommand(CallsiteDescr cd) {
        {
            std::unique_lock<std::mutex> lock(m);
            queue.push(cd);
        }
        cv.notify_all();
    }

    void loadValue(std::string name, c10::IValue value) {
        // TODO: recursive descent and move all tensors
        // isn't an issue in the prototype
        if (value.isTensor()) {
            value = c10::IValue(std::move(value).toTensor().to(device_str_, /*non_blocking=*/true));
        }
        execCommand({std::move(name), std::move(value)});
    }

    c10::IValue getValue(const std::string& name) {
        if (return_value) {
            throw std::runtime_error("Already fetching reutrn value!");
        }
        execCommand({1, name});
        std::unique_lock<std::mutex> lock(value_m);
        value_cv.wait(lock, [this, name](){return return_value;});
        auto rv = *return_value;
        return_value = c10::nullopt;
        return rv;
    }

    void deleteValue(std::string name) {
        execCommand({std::move(name)});
    }

    ~ThreadedCUDAInstance() {
        execCommand({true});
        worker_thread_.join();
    }

  private:

    int64_t device_id;
    std::mutex m;
    std::condition_variable cv;
    std::queue<CallsiteDescr> queue;
    std::unordered_map<std::string, c10::IValue> env;

    std::thread worker_thread_;
    std::string device_str_;

    std::mutex value_m;
    std::condition_variable value_cv;
    c10::optional<c10::IValue> return_value;

    at::ThreadLocalState tls_;
};

void initSPMDRuntimeBindings(PyObject* module) {
    auto m = py::handle(module).cast<py::module>();

    py::class_<CallsiteDescr>(m, "CallsiteDescr")
        .def(py::init<std::string, std::string, py::tuple, py::dict, std::string>());

    py::class_<ThreadedCUDAInstance>(m, "ThreadedCUDAInstance")
        .def(py::init<int64_t>())
        .def("exec_command", &ThreadedCUDAInstance::execCommand)
        .def("load_value", &ThreadedCUDAInstance::loadValue)
        .def("get_value", &ThreadedCUDAInstance::getValue)
        .def("delete_value", &ThreadedCUDAInstance::deleteValue);

}

}  // namespace jit
}  // namespace torch
