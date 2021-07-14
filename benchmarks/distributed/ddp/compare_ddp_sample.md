Sample Output for compare_ddp.py

Key takeaway:
- forward performance is almost the same. DDPCore is the slowest.
- async PythonDDP is slightly slower than DDPCore. DDP_CPP_CORE > PYTHON_DDP_ASYNC > PYTHON_DDP_SYNC > LEGACY_DISTRIBUTED_DATA_PARALLEL

```
Metrics for GPU 0 ddp_option DDPOption.DDP_CPP_CORE:
 1000 iterations, forward, mean=102.34264287567139 ms, median=102.1154556274414 ms, p90=102.62672271728516 ms, p99=104.59679740905761 ms
 1000 iterations, backward, mean=213.35209669494628 ms, median=213.34579467773438 ms, p90=214.5798095703125 ms, p99=216.13687896728516 ms

Metrics for GPU 0 ddp_option DDPOption.PYTHON_DDP_ASYNC:
 1000 iterations, forward, mean=100.05966097259521 ms, median=99.81171035766602 ms, p90=99.94545440673828 ms, p99=100.22951667785644 ms
 1000 iterations, backward, mean=224.905267288208 ms, median=222.97035217285156 ms, p90=224.5776107788086 ms, p99=229.21659225463867 ms

Metrics for GPU 0 ddp_option DDPOption.PYTHON_DDP_SYNC:
 1000 iterations, forward, mean=100.34294827270507 ms, median=100.10100936889648 ms, p90=100.21437454223633 ms, p99=100.3021647644043 ms
 1000 iterations, backward, mean=307.5523064880371 ms, median=305.7024383544922 ms, p90=311.60914611816406 ms, p99=316.56242706298826 ms

Metrics for GPU 0 ddp_option DDPOption.LEGACY_DISTRIBUTED_DATA_PARALLEL:
 1000 iterations, forward, mean=99.91075511932372 ms, median=99.72947311401367 ms, p90=99.83047866821289 ms, p99=99.96760696411133 ms
 1000 iterations, backward, mean=335.20290911865237 ms, median=334.42042541503906 ms, p90=340.57578735351564 ms, p99=345.0920111083984 ms
```
