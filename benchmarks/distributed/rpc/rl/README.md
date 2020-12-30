# Distributed RPC Benchmark

This tool is used to measure pytorch distributed rpc throughput and latency. This
is helpful for evaluating the suitability of rpc to applications using pytorch distributed.

In addition to printing measurements, it produces a JSON file.  Users may choose a single argument to provide multiple comma separated entries for (ie: `world_size="10,50,100"`) in which case the JSON file produced can be passed to the plotting repo to visually see how results differ.  In this case, each entry for the variable argument will be placed on the x axis.

The benchmark results comprise of 4 key metrics:
1. _Agent Latency_ - How long does it take from the time the first `select_action` request in a batch is received from an observer to the time an action is selected by the agent for each request in that batch.  If not using batch, you can think of it as `batch_size=1`.
2. _Agent Throughput_ - The number of request processed per second for a given batch.  Is literally computed as `(batch_size / agent_latency)`.  If not using batch, you can think of it as `batch_size=1`.
3. _Observer Latency_ - Time it takes from the moment a `select_action` request is sent out from a single observer to the time the response is received from the agent.  Therefore if not using batch, observer latency is `(agent_latency + communication_between_observer_and_agent_latency)`.  When using batch there will be more variation due to some observer requests being queued in a batch for longer than others depending on what order those requests came into the batch in.
4. _Observer Throughput_ - Number of requests processed per second for a single observer.  Is literally computed as `(1 / observer_latency)`.  

## Requirements

This benchmark depends on PyTorch.

## How to run

For any environments you are interested in, pass the corresponding arguments to `python launcher.py`.  The following example displays default arguments

`python launcher.py -world_size="80" -master_addr="127.0.0.1" -master_port="29501 -batch="True" -state_size="10-20-10" -nlayers="5" -out_features="10" -output_file_path="benchmark_report.json"`

Example Output:

```
---------------------------------------
PyTorch distributed rpc benchmark suite
---------------------------------------
master_addr : 127.0.0.1
master_port : 29501
batch : True
state_size : 10-20-10
nlayers : 5
out_features : 10
output_file_path : benchmark_report.json
x_axis_name : world_size
---------
Benchmark
world_size : 10
agent latency -- {50: 0.002, 75: 0.002, 90: 0.002, 95: 0.002}

agent throughput -- {50: 4071, 75: 4374, 90: 4579, 95: 4686}

observer latency -- {50: 0.003, 75: 0.003, 90: 0.003, 95: 0.003}

observer throughput -- {50: 370, 75: 386, 90: 397, 95: 404}

---------
Benchmark
world_size : 20
agent latency -- {50: 0.005, 75: 0.005, 90: 0.005, 95: 0.005}

agent throughput -- {50: 3824, 75: 4124, 90: 4471, 95: 4697}

observer latency -- {50: 0.006, 75: 0.006, 90: 0.006, 95: 0.007}

observer throughput -- {50: 176, 75: 187, 90: 198, 95: 202}

```

See plot repo for how to run plotting of output file:

![Alt text](graphs_rpc_benchmark.png?raw=true "Rpc Benchmark Plots")
