## Inference benchmarks

This folder contains a work in progress simulation of a python inference server.

The v0 version of this has a backend worker that is a single process. It loads a
ResNet-18 checkpoint to 'cuda:0' and compiles the model. It accepts requests in
the form of (tensor, request_time) from a `multiprocessing.Queue`, runs
inference on the request and returns (output, request_time) in the a separate
response `multiprocessing.Queue`.

The frontend worker is a process with three threads
1. A thread that generates fake data of a given batch size in the form of CPU
   tensors and puts the data into the request queue
2. A thread that reads responses from the response queue and collects metrics on
   the latency of the first response, which corresponds to the cold start time,
   average, minimum and maximum response latency as well as throughput.
3. A thread that polls nvidia-smi for GPU utilization metrics.

For now we omit data preprocessing as well as result post-processing.

### Running a single benchmark

The togglable commmand line arguments to the script are as follows:
  - `num_iters` (default: 100): how many requests to send to the backend
    excluding the first warmup request
  - `batch_size` (default: 32): the batch size of the requests.
  - `model_dir` (default: '.'): the directory to load the checkpoint from
  - `compile` (default: compile): or `--no-compile` whether to `torch.compile()`
    the model
  - `output_file` (default: output.csv): The name of the csv file to write the outputs to in the `results/` directory.
  - `num_workers` (default: 2): The `max_threads` passed to the `ThreadPoolExecutor` in charge of model prediction

e.g. A sample command to run the benchmark

```
python -W ignore server.py --num_iters 1000 --batch_size 32
```

the results will be found in `results/output.csv`, which will be appended to if the file already exists.

Note that `m.compile()` time in the csv file is not the time for the model to be compiled,
which happens during the first iteration, but rather the time for PT2 components
to be lazily imported (e.g. triton).

### Running a sweep

The script `runner.sh` will run a sweep of the benchmark over different batch
sizes with compile on and off and collect the mean and standard deviation of warmup latency,
average latency, throughput and GPU utilization for each. The `results/` directory will contain the metrics
from running a sweep as we develop this benchmark where `results/output_{batch_size}_{compile}.md`
will contain the mean and standard deviation of results for a given batch size and compile setting.
If the file already exists, the metrics from the run will be appended as a new row in the markdown table.
