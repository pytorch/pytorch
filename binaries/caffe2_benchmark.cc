#include <fstream>
#include <iterator>
#include <string>

#include "binaries/benchmark_args.h"
#include "binaries/benchmark_helper.h"


int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  benchmark(
      argc,
      argv,
      FLAGS_backend,
      FLAGS_init_net,
      FLAGS_input,
      FLAGS_input_dims,
      FLAGS_input_file,
      FLAGS_input_type,
      FLAGS_iter,
      FLAGS_measure_memory,
      FLAGS_net,
      FLAGS_output,
      FLAGS_output_folder,
      FLAGS_run_individual,
      FLAGS_sleep_before_run,
      FLAGS_sleep_between_iteration,
      FLAGS_sleep_between_net_and_operator,
      FLAGS_text_output,
      FLAGS_warmup,
      FLAGS_wipe_cache);
}
