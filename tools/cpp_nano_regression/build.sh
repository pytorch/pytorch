#!/bin/sh
g++ -o tracker.so -shared -fPIC tracker.cpp
g++ -o printing_tracker.so -shared -fPIC printing_tracker.cpp
g++ -o test_lib.so -shared -fPIC --std=c++11 --include=prelude.h test_lib.cpp
g++ -o test -L`pwd` -Wl,-R`pwd` test_lib.so test.cpp
