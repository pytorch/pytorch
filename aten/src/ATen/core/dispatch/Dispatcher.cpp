#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

namespace detail {
class RegistrationListenerList final {
public:
  void addListener(std::unique_ptr<OpRegistrationListener> listener) {
    listeners_.push_back(std::move(listener));
  }

  void callOnOperatorRegistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorRegistered(op);
    }
  }

  void callOnOperatorDeregistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorDeregistered(op);
    }
  }
private:
  std::vector<std::unique_ptr<OpRegistrationListener>> listeners_;
};
}

OpRegistrationListener::~OpRegistrationListener() {}

Dispatcher::Dispatcher()
: operators_()
, listeners_(guts::make_unique<detail::RegistrationListenerList>())
, mutex_() {}

Dispatcher::~Dispatcher() {}

C10_EXPORT Dispatcher& Dispatcher::singleton() {
  static Dispatcher _singleton;
  return _singleton;
}

namespace {
class SchemaEqualsCheck final {
public:
  SchemaEqualsCheck(const FunctionSchema& expected, const FunctionSchema& actual)
  : expected_(expected), actual_(actual) {}

  void check() {
    // this function should only be called when name and overload_name are already equal.
    AT_ASSERT(expected_.name() == actual_.name());
    AT_ASSERT(expected_.overload_name() == actual_.overload_name());

    if (!argumentsEqual(expected_.arguments(), actual_.arguments())) {
      reportSchemaNotEquals("they have different arguments");
    }
    if (!argumentsEqual(expected_.returns(), actual_.returns())) {
      reportSchemaNotEquals("they have different returns");
    }
    if (expected_.is_vararg() != actual_.is_vararg()) {
      reportSchemaNotEquals("one of them has varargs and one doesn't");
    }
    if (expected_.is_vararg() != actual_.is_vararg()) {
      reportSchemaNotEquals("one of them has variable returns and one doesn't");
    }
  }

private:
  C10_NORETURN void reportSchemaNotEquals(const std::string& message) {
      throw std::logic_error("Tried to register multiple operator schemas for " + expected_.name() + "." + expected_.overload_name() + ", but: " + message);
  }

  bool argumentsEqual(const std::vector<Argument>& expected, const std::vector<Argument>& actual) {
    if(expected.size() != actual.size()) {
      return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
      if (!argumentEquals(expected[i], actual[i])) {
        return false;
      }
    }

    return true;
  }

  bool argumentEquals(const Argument& expected, const Argument& actual) {
    return expected.name() == actual.name()
        && expected.type() == actual.type()
        && expected.N() == actual.N()
        // TODO && expected.default_value() == actual.default_value()
        && expected.kwarg_only() == actual.kwarg_only()
        ;// TODO && expected.alias_info() == actual.alias_info();
  }

  const FunctionSchema& expected_;
  const FunctionSchema& actual_;
};

}

std::list<Dispatcher::OperatorDef>::iterator Dispatcher::findOrRegisterSchema_(FunctionSchema&& schema) {
  const auto found = std::find_if(operators_.begin(), operators_.end(), [&] (const OperatorDef& opDef) {
    return opDef.schema.name() == schema.name() && opDef.schema.overload_name() == schema.overload_name();
  });
  if (found != operators_.end()) {
    SchemaEqualsCheck(found->schema, schema).check();
    return found;
  }

  operators_.emplace_back(std::move(schema));
  return --operators_.end();
}

OperatorHandle Dispatcher::registerSchema(FunctionSchema schema) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = OperatorHandle(findOrRegisterSchema_(std::move(schema)));

  ++op.operatorDefIterator_->refcount;
  if (1 == op.operatorDefIterator_->refcount) {
    // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
    listeners_->callOnOperatorRegistered(op);
  }

  return op;
}

void Dispatcher::deregisterSchema(const OperatorHandle& op) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  if (!op.operatorDefIterator_->dispatchTable.isEmpty()) {
    AT_ERROR("Tried to deregister op schema that still has kernels registered");
  }

  // reduce refcount and actually deregister if no references left
  --op.operatorDefIterator_->refcount;
  if (0 == --op.operatorDefIterator_->refcount) {
    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    listeners_->callOnOperatorDeregistered(op);

    operators_.erase(op.operatorDefIterator_);
  }
}

void Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, cache_creator_func});
}

void Dispatcher::deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.deregisterKernel(dispatch_key);
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

}
