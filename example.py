from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops

schedule_ops = get_schedule_ops(
    schedule="Interleaved1F1B", pp_degree=2, num_microbatches=4, with_comms=True
)
for row in schedule_ops:
    print(row)
