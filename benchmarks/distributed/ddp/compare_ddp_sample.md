Sample Output for compare_ddp.py

Key takeaway:
- forward performance is almost the same. DDPCore is the slowest.
- async PythonDDP is slightly slower than DDPCore. DDP_CPP_CORE > PYTHON_DDP_ASYNC > PYTHON_DDP_SYNC > LEGACY_DISTRIBUTED_DATA_PARALLEL

```
Metrics for GPU 0 ddp_option DDPOption.DDP_CPP_CORE:
 100 iterations, forward, mean=103.39265975952148 ms, median=101.62957000732422 ms, p90=101.9048828125 ms, p99=104.3886895751962 ms
 100 iterations, backward, mean=212.88988677978514 ms, median=212.41372680664062 ms, p90=213.5197265625 ms, p99=215.54352249145538 ms

 Metrics for GPU 0 ddp_option DDPOption.PYTHON_DDP_ASYNC:
 100 iterations, forward, mean=102.17666648864746 ms, median=99.81665802001953 ms, p90=99.97232055664062 ms, p99=102.46604179382444 ms
 100 iterations, backward, mean=240.18947006225585 ms, median=221.8608627319336 ms, p90=223.41331329345704 ms, p99=243.52904235840776 ms

Metrics for GPU 0 ddp_option DDPOption.PYTHON_DDP_SYNC:
 100 iterations, forward, mean=101.82003700256348 ms, median=99.88966369628906 ms, p90=100.03099517822265 ms, p99=102.35179565429786 ms
 100 iterations, backward, mean=314.58491973876954 ms, median=297.79063415527344 ms, p90=305.7146484375 ms, p99=328.67885314942254 ms

Metrics for GPU 0 ddp_option DDPOption.LEGACY_DISTRIBUTED_DATA_PARALLEL:
 100 iterations, forward, mean=102.28044731140136 ms, median=100.01128005981445 ms, p90=100.10251693725586 ms, p99=102.46430564880487 ms
 100 iterations, backward, mean=353.54674926757815 ms, median=336.01605224609375 ms, p90=339.9256134033203 ms, p99=362.1843984985446 ms
```
