# Try to find the Google Benchmark library and headers.
#  Benchmark_FOUND        - system has benchmark lib
#  Benchmark_INCLUDE_DIRS - the benchmark include directory
#  Benchmark_LIBRARIES    - libraries needed to use benchmark

find_path(Benchmark_INCLUDE_DIR
	NAMES benchmark/benchmark.h
	DOC "The directory where benchmark includes reside"
)

find_library(Benchmark_LIBRARY
	NAMES benchmark
	DOC "The benchmark library"
)

set(Benchmark_INCLUDE_DIRS ${Benchmark_INCLUDE_DIR})
set(Benchmark_LIBRARIES    ${Benchmark_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Benchmark
	FOUND_VAR Benchmark_FOUND
	REQUIRED_VARS Benchmark_INCLUDE_DIR Benchmark_LIBRARY
)

mark_as_advanced(Benchmark_FOUND)
