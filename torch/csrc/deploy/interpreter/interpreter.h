#pragma once
#include <dlfcn.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>


class Interpreter : public InterpreterImpl {
 private:
  std::string library_name_;
  void* handle_;

 public:
  Interpreter() : handle_(nullptr) {
    char library_name[L_tmpnam];
    library_name_ = library_name;
    char* libinterpreter_path = std::getenv("LIBINTERPRETER_PATH");
    if (libinterpreter_path == nullptr) {
      throw std::runtime_error("libinterpreter_path is NULL, set LIBINTERPRETER_PATH env.");
    }
    std::tmpnam(library_name);
    {
      std::ifstream src(libinterpreter_path,  std::ios::binary);
      std::ofstream dst(library_name, std::ios::binary);
      dst << src.rdbuf();
    }
    handle_ = dlopen(library_name, RTLD_LOCAL | RTLD_LAZY);
    if (!handle_) {
      throw std::runtime_error(dlerror());
    }

    // technically, we can unlike the library right after dlopen, and this is
    // better for cleanup because even if we crash the library doesn't stick
    // around. However, its crap for debugging because gdb can't find the
    // symbols if the library is no longer present.
    unlink(library_name_.c_str());

    void* initialize_interface = dlsym(handle_, "initialize_interface");
    if (!initialize_interface) {
      throw std::runtime_error("Unable to load initialize_interface function from interpreter lib.");
    }
    ((void (*)(InterpreterImpl*))initialize_interface)(this);

    this->startup();

    // the actual torch loading process is not thread safe, by doing it
    // in the constructor before we have multiple worker threads, then we
    // ensure it doesn't race.
    run_some_python("import torch");
  }
  ~Interpreter() {
    if (handle_) {
      this->teardown();

      // it segfaults its face off trying to unload, but it's not clear
      // if this is something we caused of if libtorch_python would also do the
      // same if it were opened/closed a lot...
      dlclose(handle_);
    }
  }
  Interpreter(const Interpreter&) = delete;
};
