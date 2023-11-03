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
3. A thread that polls nvidia-smi for GPU utilization metrics every 100ms.

For now we omit data preprocessing as well as result post-processing.

### Running a single benchmark

The togglable commmand line arguments to the script are as follows:
  - `num_iters` (default: 100): how many requests to send to the backend
    excluding the first warmup request
  - `batch_size` (default: 32): the batch size of the requests.
  - `model_dir` (default: '.'): the directory to load the checkpoint from
  - `compile` (default: True): whether to `torch.compile()` the model

e.g. A sample command to run the benchmark

```
python -W ignore server.py --num_iters 1000 --batch_size 32
```

A sample output is

```
torch.load() time: 3.95351 s
m.compile() time (not actual first compilation): 3.41085 s
Warmup latency: 15.92736 s
Average latency (exclude warmup): 0.09556 +/- 0.07029 s
Max latency: 0.60715 s
Min latency: 0.05200 s
Throughput (exclude warmup): 334.85437 samples per second
Average GPU utilization: 20.74092
```

Note that `m.compile()` time above is not the time for the model to be compiled,
which happens during the first iteration, but rather the time for PT2 components
to be lazily imported (e.g. triton).

### Running a sweep

The script `runner.sh` will run a sweep of the benchmark over different batch
sizes with compile on and off. The `results/` directory will contain the metrics
from running a sweep as we develop this benchmark.
