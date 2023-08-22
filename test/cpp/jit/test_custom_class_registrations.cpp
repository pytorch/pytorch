#include <test/cpp/jit/test_custom_class_registrations.h>

#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

using namespace torch::jit;

namespace {

struct DefaultArgs : torch::CustomClassHolder {
  int x;
  DefaultArgs(int64_t start = 3) : x(start) {}
  int64_t increment(int64_t val = 1) {
    x += val;
    return x;
  }
  int64_t decrement(int64_t val = 1) {
    x += val;
    return x;
  }
  int64_t scale_add(int64_t add, int64_t scale = 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    x = scale * x + add;
    return x;
  }
  int64_t divide(c10::optional<int64_t> factor) {
    if (factor) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      x = x / *factor;
    }
    return x;
  }
};

struct Foo : torch::CustomClassHolder {
  int x, y;
  Foo() : x(0), y(0) {}
  Foo(int x_, int y_) : x(x_), y(y_) {}
  int64_t info() {
    return this->x * this->y;
  }
  int64_t add(int64_t z) {
    return (x + y) * z;
  }
  void increment(int64_t z) {
    this->x += z;
    this->y += z;
  }
  int64_t combine(c10::intrusive_ptr<Foo> b) {
    return this->info() + b->info();
  }
};

struct _StaticMethod : torch::CustomClassHolder {
  // NOLINTNEXTLINE(modernize-use-equals-default)
  _StaticMethod() {}
  static int64_t staticMethod(int64_t input) {
    return 2 * input;
  }
};

struct FooGetterSetter : torch::CustomClassHolder {
  FooGetterSetter() : x(0), y(0) {}
  FooGetterSetter(int64_t x_, int64_t y_) : x(x_), y(y_) {}

  int64_t getX() {
    // to make sure this is not just attribute lookup
    return x + 2;
  }
  void setX(int64_t z) {
    // to make sure this is not just attribute lookup
    x = z + 2;
  }

  int64_t getY() {
    // to make sure this is not just attribute lookup
    return y + 4;
  }

 private:
  int64_t x, y;
};

struct FooGetterSetterLambda : torch::CustomClassHolder {
  int64_t x;
  FooGetterSetterLambda() : x(0) {}
  FooGetterSetterLambda(int64_t x_) : x(x_) {}
};

struct FooReadWrite : torch::CustomClassHolder {
  int64_t x;
  const int64_t y;
  FooReadWrite() : x(0), y(0) {}
  FooReadWrite(int64_t x_, int64_t y_) : x(x_), y(y_) {}
};

struct LambdaInit : torch::CustomClassHolder {
  int x, y;
  LambdaInit(int x_, int y_) : x(x_), y(y_) {}
  int64_t diff() {
    return this->x - this->y;
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct NoInit : torch::CustomClassHolder {
  int64_t x;
};

struct PickleTester : torch::CustomClassHolder {
  PickleTester(std::vector<int64_t> vals) : vals(std::move(vals)) {}
  std::vector<int64_t> vals;
};

at::Tensor take_an_instance(const c10::intrusive_ptr<PickleTester>& instance) {
  return torch::zeros({instance->vals.back(), 4});
}

struct ElementwiseInterpreter : torch::CustomClassHolder {
  using InstructionType = std::tuple<
      std::string /*op*/,
      std::vector<std::string> /*inputs*/,
      std::string /*output*/>;

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ElementwiseInterpreter() {}

  // Load a list of instructions into the interpreter. As specified above,
  // instructions specify the operation (currently support "add" and "mul"),
  // the names of the input values, and the name of the single output value
  // from this instruction
  void setInstructions(std::vector<InstructionType> instructions) {
    instructions_ = std::move(instructions);
  }

  // Add a constant. The interpreter maintains a set of constants across
  // calls. They are keyed by name, and constants can be referenced in
  // Instructions by the name specified
  void addConstant(const std::string& name, at::Tensor value) {
    constants_.insert_or_assign(name, std::move(value));
  }

  // Set the string names for the positional inputs to the function this
  // interpreter represents. When invoked, the interpreter will assign
  // the positional inputs to the names in the corresponding position in
  // input_names.
  void setInputNames(std::vector<std::string> input_names) {
    input_names_ = std::move(input_names);
  }

  // Specify the output name for the function this interpreter represents. This
  // should match the "output" field of one of the instructions in the
  // instruction list, typically the last instruction.
  void setOutputName(std::string output_name) {
    output_name_ = std::move(output_name);
  }

  // Invoke this interpreter. This takes a list of positional inputs and returns
  // a single output. Currently, inputs and outputs must all be Tensors.
  at::Tensor __call__(std::vector<at::Tensor> inputs) {
    // Environment to hold local variables
    std::unordered_map<std::string, at::Tensor> environment;

    // Load inputs according to the specified names
    if (inputs.size() != input_names_.size()) {
      std::stringstream err;
      err << "Expected " << input_names_.size() << " inputs, but got "
          << inputs.size() << "!";
      throw std::runtime_error(err.str());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      environment[input_names_[i]] = inputs[i];
    }

    for (InstructionType& instr : instructions_) {
      // Retrieve all input values for this op
      std::vector<at::Tensor> inputs;
      for (const auto& input_name : std::get<1>(instr)) {
        // Operator output values shadow constants.
        // Imagine all constants are defined in statements at the beginning
        // of a function (a la K&R C). Any definition of an output value must
        // necessarily come after constant definition in textual order. Thus,
        // We look up values in the environment first then the constant table
        // second to implement this shadowing behavior
        if (environment.find(input_name) != environment.end()) {
          inputs.push_back(environment.at(input_name));
        } else if (constants_.find(input_name) != constants_.end()) {
          inputs.push_back(constants_.at(input_name));
        } else {
          std::stringstream err;
          err << "Instruction referenced unknown value " << input_name << "!";
          throw std::runtime_error(err.str());
        }
      }

      // Run the specified operation
      at::Tensor result;
      const auto& op = std::get<0>(instr);
      if (op == "add") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for add op!");
        }
        result = inputs[0] + inputs[1];
      } else if (op == "mul") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for mul op!");
        }
        result = inputs[0] * inputs[1];
      } else {
        std::stringstream err;
        err << "Unknown operator " << op << "!";
        throw std::runtime_error(err.str());
      }

      // Write back result into environment
      const auto& output_name = std::get<2>(instr);
      environment[output_name] = std::move(result);
    }

    if (!output_name_) {
      throw std::runtime_error("Output name not specified!");
    }

    return environment.at(*output_name_);
  }

  // Ser/De infrastructure. See
  // https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html#defining-serialization-deserialization-methods-for-custom-c-classes
  // for more info.

  // This is the type we will use to marshall information on disk during
  // ser/de. It is a simple tuple composed of primitive types and simple
  // collection types like vector, optional, and dict.
  using SerializationType = std::tuple<
      std::vector<std::string> /*input_names_*/,
      c10::optional<std::string> /*output_name_*/,
      c10::Dict<std::string, at::Tensor> /*constants_*/,
      std::vector<InstructionType> /*instructions_*/
      >;

  // This function yields the SerializationType instance for `this`.
  SerializationType __getstate__() const {
    return SerializationType{
        input_names_, output_name_, constants_, instructions_};
  }

  // This function will create an instance of `ElementwiseInterpreter` given
  // an instance of `SerializationType`.
  static c10::intrusive_ptr<ElementwiseInterpreter> __setstate__(
      SerializationType state) {
    auto instance = c10::make_intrusive<ElementwiseInterpreter>();
    std::tie(
        instance->input_names_,
        instance->output_name_,
        instance->constants_,
        instance->instructions_) = std::move(state);
    return instance;
  }

  // Class members
  std::vector<std::string> input_names_;
  c10::optional<std::string> output_name_;
  c10::Dict<std::string, at::Tensor> constants_;
  std::vector<InstructionType> instructions_;
};

struct ReLUClass : public torch::CustomClassHolder {
  at::Tensor run(const at::Tensor& t) {
    return t.relu();
  }
};

TORCH_LIBRARY(_TorchScriptTesting, m) {
  m.class_<ScalarTypeClass>("_ScalarTypeClass")
      .def(torch::init<at::ScalarType>())
      .def_pickle(
          [](const c10::intrusive_ptr<ScalarTypeClass>& self) {
            return std::make_tuple(self->scalar_type_);
          },
          [](std::tuple<at::ScalarType> s) {
            return c10::make_intrusive<ScalarTypeClass>(std::get<0>(s));
          });

  m.class_<ReLUClass>("_ReLUClass")
      .def(torch::init<>())
      .def("run", &ReLUClass::run);

  m.class_<_StaticMethod>("_StaticMethod")
      .def(torch::init<>())
      .def_static("staticMethod", &_StaticMethod::staticMethod);

  m.class_<DefaultArgs>("_DefaultArgs")
      .def(torch::init<int64_t>(), "", {torch::arg("start") = 3})
      .def("increment", &DefaultArgs::increment, "", {torch::arg("val") = 1})
      .def("decrement", &DefaultArgs::decrement, "", {torch::arg("val") = 1})
      .def(
          "scale_add",
          &DefaultArgs::scale_add,
          "",
          {torch::arg("add"), torch::arg("scale") = 1})
      .def(
          "divide",
          &DefaultArgs::divide,
          "",
          {torch::arg("factor") = torch::arg::none()});

  m.class_<Foo>("_Foo")
      .def(torch::init<int64_t, int64_t>())
      // .def(torch::init<>())
      .def("info", &Foo::info)
      .def("increment", &Foo::increment)
      .def("add", &Foo::add)
      .def("combine", &Foo::combine);

  m.class_<FooGetterSetter>("_FooGetterSetter")
      .def(torch::init<int64_t, int64_t>())
      .def_property("x", &FooGetterSetter::getX, &FooGetterSetter::setX)
      .def_property("y", &FooGetterSetter::getY);

  m.class_<FooGetterSetterLambda>("_FooGetterSetterLambda")
      .def(torch::init<int64_t>())
      .def_property(
          "x",
          [](const c10::intrusive_ptr<FooGetterSetterLambda>& self) {
            return self->x;
          },
          [](const c10::intrusive_ptr<FooGetterSetterLambda>& self,
             int64_t val) { self->x = val; });

  m.class_<FooReadWrite>("_FooReadWrite")
      .def(torch::init<int64_t, int64_t>())
      .def_readwrite("x", &FooReadWrite::x)
      .def_readonly("y", &FooReadWrite::y);

  m.class_<LambdaInit>("_LambdaInit")
      .def(torch::init([](int64_t x, int64_t y, bool swap) {
        if (swap) {
          return c10::make_intrusive<LambdaInit>(y, x);
        } else {
          return c10::make_intrusive<LambdaInit>(x, y);
        }
      }))
      .def("diff", &LambdaInit::diff);

  m.class_<NoInit>("_NoInit").def(
      "get_x", [](const c10::intrusive_ptr<NoInit>& self) { return self->x; });

  m.class_<MyStackClass<std::string>>("_StackString")
      .def(torch::init<std::vector<std::string>>())
      .def("push", &MyStackClass<std::string>::push)
      .def("pop", &MyStackClass<std::string>::pop)
      .def("clone", &MyStackClass<std::string>::clone)
      .def("merge", &MyStackClass<std::string>::merge)
      .def_pickle(
          [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
            return self->stack_;
          },
          [](std::vector<std::string> state) { // __setstate__
            return c10::make_intrusive<MyStackClass<std::string>>(
                std::vector<std::string>{"i", "was", "deserialized"});
          })
      .def("return_a_tuple", &MyStackClass<std::string>::return_a_tuple)
      .def(
          "top",
          [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
              -> std::string { return self->stack_.back(); })
      .def(
          "__str__",
          [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < self->stack_.size(); ++i) {
              ss << self->stack_[i];
              if (i != self->stack_.size() - 1) {
                ss << ", ";
              }
            }
            ss << "]";
            return ss.str();
          });
  // clang-format off
        // The following will fail with a static assert telling you you have to
        // take an intrusive_ptr<MyStackClass> as the first argument.
        // .def("foo", [](int64_t a) -> int64_t{ return 3;});
  // clang-format on

  m.class_<PickleTester>("_PickleTester")
      .def(torch::init<std::vector<int64_t>>())
      .def_pickle(
          [](c10::intrusive_ptr<PickleTester> self) { // __getstate__
            return self->vals;
          },
          [](std::vector<int64_t> state) { // __setstate__
            return c10::make_intrusive<PickleTester>(std::move(state));
          })
      .def(
          "top",
          [](const c10::intrusive_ptr<PickleTester>& self) {
            return self->vals.back();
          })
      .def("pop", [](const c10::intrusive_ptr<PickleTester>& self) {
        auto val = self->vals.back();
        self->vals.pop_back();
        return val;
      });

  m.def(
      "take_an_instance(__torch__.torch.classes._TorchScriptTesting._PickleTester x) -> Tensor Y",
      take_an_instance);
  // test that schema inference is ok too
  m.def("take_an_instance_inferred", take_an_instance);

  m.class_<ElementwiseInterpreter>("_ElementwiseInterpreter")
      .def(torch::init<>())
      .def("set_instructions", &ElementwiseInterpreter::setInstructions)
      .def("add_constant", &ElementwiseInterpreter::addConstant)
      .def("set_input_names", &ElementwiseInterpreter::setInputNames)
      .def("set_output_name", &ElementwiseInterpreter::setOutputName)
      .def("__call__", &ElementwiseInterpreter::__call__)
      .def_pickle(
          /* __getstate__ */
          [](const c10::intrusive_ptr<ElementwiseInterpreter>& self) {
            return self->__getstate__();
          },
          /* __setstate__ */
          [](ElementwiseInterpreter::SerializationType state) {
            return ElementwiseInterpreter::__setstate__(std::move(state));
          });
}

} // namespace
