#include <future>
#include <iostream>
#include "interpreter.h"

int main() {
  Interpreter interp;
  interp.run_some_python("print('hello from first interpeter!')");
}