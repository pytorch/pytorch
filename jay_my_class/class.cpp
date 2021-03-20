// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>

// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string>
#include <vector>

template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }

  T pop() {
    
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }
};


// Notice a few things:
// - We pass the class to be registered as a template parameter to
//   `torch::class_`. In this instance, we've passed the
//   specialization of the MyStackClass class ``MyStackClass<std::string>``.
//   In general, you cannot register a non-specialized template
//   class. For non-templated classes, you can just pass the
//   class name directly as the template parameter.
// - The arguments passed to the constructor make up the "qualified name"
//   of the class. In this case, the registered class will appear in
//   Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
//   the first argument the "namespace" and the second argument the
//   actual class name.
TORCH_LIBRARY(my_classes, m) {
//   m.class_<MyStackClass<std::string>>("MyStackClass")
m.class_<MyStackClass<std::string>>("MyStackClass")
    // The following line registers the contructor of our MyStackClass
    // class that takes a single `std::vector<std::string>` argument,
    // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
    // Currently, we do not support registering overloaded
    // constructors, so for now you can only `def()` one instance of
    // `torch::init`.
    .def(torch::init<std::vector<std::string>>())
    // The next line registers a stateless (i.e. no captures) C++ lambda
    // function as a method. Note that a lambda function must take a
    // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // as the first argument. Other arguments can be whatever you want.
    .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
      return self->stack_.back();
    })
    // The following four lines expose methods of the MyStackClass<std::string>
    // class as-is. `torch::class_` will automatically examine the
    // argument and return types of the passed-in method pointers and
    // expose these to Python and TorchScript accordingly. Finally, notice
    // that we must take the *address* of the fully-qualified method name,
    // i.e. use the unary `&` operator, due to C++ typing rules.
    .def("push", &MyStackClass<std::string>::push)
    .def("pop", &MyStackClass<std::string>::pop)
    .def("clone", &MyStackClass<std::string>::clone)
    .def("merge", &MyStackClass<std::string>::merge)
  ;
}