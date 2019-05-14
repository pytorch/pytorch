#pragma once

#include <c10/core/Backend.h>

namespace at {

template <typename Return, typename ...Parameters>
class ATenOperator final {
 public:
   explicit ATenOperator(Return (*op)(Parameters...), Return (*autograd_wrapper)(Return (*)(Parameters...), Parameters...))
   : op_(op), autograd_wrapper_(autograd_wrapper) {}

  Return operator()(Parameters... params) const {
   if (autograd_wrapper_) {
     return (*autograd_wrapper_)(op_, params...);
   }
   return (*op_)(params...);
  }

 private:
  Return (*op_)(Parameters...);
  Return (*autograd_wrapper_)(Return (*)(Parameters...), Parameters...);
};

int64_t _register_op(Backend backend, const char* schema, void* fn);

int64_t _register_variable_wrapper(const char* schema, void* fn);

template <typename FnPtr>
int64_t register_op(Backend backend, const char* schema, FnPtr fn) {
  return _register_op(backend, schema, reinterpret_cast<void*>(fn));
}

template <typename FnPtr>
int64_t register_variable_wrapper(const char* schema, FnPtr fn) {
  return _register_variable_wrapper(schema, reinterpret_cast<void*>(fn));
}

void* get_op(Backend backend, int64_t id);

void* get_variable_wrapper(int64_t id);

template <typename Return, typename ...Parameters>
static ATenOperator<Return, Parameters...> find_op(Backend backend, bool is_variable, int64_t id) {
  auto op = reinterpret_cast<Return (*)(Parameters...)>(get_op(backend, id));
  auto op_wrapper = reinterpret_cast<Return (*)(Return (*)(Parameters...), Parameters...)>(get_variable_wrapper(id));
  if (is_variable) {
    return ATenOperator<Return, Parameters...>(op, op_wrapper);
  }
  return ATenOperator<Return, Parameters...>(op, nullptr);
}
} // namespace at
