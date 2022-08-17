#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdint>
#include <cstdio>
#include <iostream>

namespace py = pybind11;

int foo_constructed = 0;
int bar_constructed = 0;
int bar_destructed = 0;

struct Foo {
  Foo() {
    ++foo_constructed;
  }
  int v = -1;
};

Foo f;

struct Bar {
  Bar() {
    ++bar_constructed;
  }
  ~Bar() {
    ++bar_destructed;
  }
  int v = 14;
};

static thread_local int first = 1; // local TLS, probably offset 0
static thread_local int second = 2; // local TLS, probably offset 4
thread_local int bss_local; // local TLS, bss initialized so it probably comes
                            // after the initialized stuff
thread_local int third = 3; // local TLS, but extern declared so it will look
                            // for the symbol third globally, but not find it
static thread_local Bar bar; // local TLS, with a constructor that should run
thread_local int
    in_another_module; // non local TLS, this is defined in test_deploy.cpp

struct MyError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

bool raise_and_catch_exception(bool except) {
  try {
    if (except) {
      throw MyError("yep");
    }
    return false;
  } catch (MyError& c) {
    return true;
  }
}
bool raise_exception() {
  throw MyError("yet"); // caught in test_deploy
}

bool check_initial_state() {
  bool bv = bar.v == 14; // unless we reference bar it is unspecified whether it
                         // should have been constructed
  return bv && first == 1 && second == 2 && bss_local == 0 && third == 3 &&
      bar_constructed == 1 && foo_constructed == 1 && bar_destructed == 0;
}

int get_in_another_module() {
  return in_another_module;
}

void set_in_another_module(int x) {
  in_another_module = x;
}
int get_bar() {
  return bar.v;
}
void set_bar(int v) {
  bar.v = v;
}
int get_bar_destructed() {
  return bar_destructed;
}

int simple_add(int a, int b) {
  return a + b;
}

PYBIND11_MODULE(libtest_deploy_lib, m) {
  m.def("raise_and_catch_exception", raise_and_catch_exception);
  m.def("raise_exception", raise_exception);
  m.def("check_initial_state", check_initial_state);
  m.def("get_in_another_module", get_in_another_module);
  m.def("set_in_another_module", set_in_another_module);
  m.def("get_bar", get_bar);
  m.def("set_bar", set_bar);
  m.def("get_bar_destructed", get_bar_destructed);
  m.def("simple_add", simple_add);
}
