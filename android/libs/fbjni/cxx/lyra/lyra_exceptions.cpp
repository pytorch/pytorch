/*
 *  Copyright (c) 2004-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include <lyra/lyra_exceptions.h>

#include <cstdlib>
#include <exception>
#include <sstream>
#include <typeinfo>

#include <fbjni/detail/Log.h>

namespace facebook {
namespace lyra {

using namespace detail;

namespace {
std::terminate_handler gTerminateHandler;

const ExceptionTraceHolder* getExceptionTraceHolder(std::exception_ptr ptr) {
  try {
    std::rethrow_exception(ptr);
  } catch (const ExceptionTraceHolder& holder) {
    return &holder;
  } catch (...) {
    return nullptr;
  }
}

void logExceptionAndAbort() {
  if (auto ptr = std::current_exception()) {
    FBJNI_LOGE("Uncaught exception: %s", toString(ptr).c_str());
    auto trace = getExceptionTraceHolder(ptr);
    if (trace) {
      logStackTrace(getStackTraceSymbols(trace->stackTrace_));
    }
  }
  if (gTerminateHandler) {
    gTerminateHandler();
  } else {
    FBJNI_LOGF("Uncaught exception and no gTerminateHandler set");
  }
}

const std::vector<InstructionPointer> emptyTrace;
} // namespace

ExceptionTraceHolder::~ExceptionTraceHolder() {}

detail::ExceptionTraceHolder::ExceptionTraceHolder() {
  // TODO(cjhopman): This should be done more safely (i.e. use preallocated space, etc.).
  stackTrace_.reserve(128);
  getStackTrace(stackTrace_, 1);
}


void ensureRegisteredTerminateHandler() {
  static auto initializer = (gTerminateHandler = std::set_terminate(logExceptionAndAbort));
  (void)initializer;
}

const std::vector<InstructionPointer>& getExceptionTrace(std::exception_ptr ptr) {
  auto holder = getExceptionTraceHolder(ptr);
  return holder ? holder->stackTrace_ : emptyTrace;
}

std::string toString(std::exception_ptr ptr) {
  if (!ptr) {
    return "No exception";
  }

  try {
    std::rethrow_exception(ptr);
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << typeid(e).name() << ": " << e.what();
    return ss.str();
  } catch (...) {
    return "Unknown exception";
  }
}

}
}
