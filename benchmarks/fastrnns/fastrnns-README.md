Some of these scripts accept command line args but most of them do not because
I was lazy. They will probably be added sometime in the future, but the default
sizes are pretty reasonable.

### Test lstm (fwd + bwd) correctness
python -m fastrnns.test

### Run some lstm benchmarking
python -m fastrnns.bench

### Run some lstm profiling. Calls nvprof.
python -m fastrnns.profile

