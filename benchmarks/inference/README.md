## Inference benchmarks

This folder contains a work in progress simulation of a python inference server.

The v0 version of this has a backend worker that is a single process. It loads a
ResNet-18 checkpoint to 'cuda:0' and compiles the model. It accepts requests in
the form of (tensor, request_time) from a `multiprocessing.Queue`, runs
inference on the request and returns (output, request_time) in the a separate
response `multiprocessing.Queue`.

The frontend worker generates fake data in the form of tensors and puts
it into the request queue. It collects metrics on the latency of the first
response, which corresponds to the cold start time, average response latency
as well as throughput.

For now we omit data preprocessing as well as result post-processing.

The togglable commmand line arguments to the script are as follows:
  - `num_iters` (default: 100): how many requests to send to the backend
    excluding the first warmup request
  - `batch_size` (default: 32): the batch size of the requests.
  - `model_dir` (default: '.'): the directory to load the checkpoint from
  - `compile` (default: True): whether to `torch.compile()` the model

e.g. A sample command to run the benchmark

```
python server.py --num_iters 1000 --batch_size 32
```

A sample output is

```
Load time: 4.50205 seconds
Compile time: 2.62402 seconds
Warmup latency: 17.88542 seconds
Average latency (exclude warmup): 0.09022 +/- 0.04824 seconds, max 0.33756 seconds, min 0.05443 seconds
Throughput (exclude warmup): 11.09536104939165 batches per second
```
