#include <future>
#include <iostream>
#include "interpreter.h"

int main() {
  Interpreter interp;
  interp.run_some_python("print('hello from first interpeter!')");

  Interpreter interp2;
  interp2.run_some_python("print('hello from second interpeter!')");
}