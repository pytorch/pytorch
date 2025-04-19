# Instructions on how to make a new compile time benchmark

1. Make a new benchmark file in /benchmarks/dynamo/pr_time_benchmarks/benchmarks/ eg. https://github.com/pytorch/pytorch/blob/0b75b7ff2b8ab8f40e433a52b06a671d6377997f/benchmarks/dynamo/pr_time_benchmarks/benchmarks/add_loop.py
2. cd into the pr_time_benchmarks directory `cd benchmarks/dynamo/pr_time_benchmarks`
3. Run `PYTHONPATH=./ python benchmarks/[YOUR_BENCHMARK].py a.txt`
4. (Optional) flip a flag that you know will change the benchmark and run again with b.txt `PYTHONPATH=./ python benchmarks/[YOUR_BENCHMARK].py a.txt`
5. Compare `a.txt` and `b.txt` located within the `benchmarks/dynamo/pr_time_benchmarks` folder to make sure things look as you expect
6. Check in your new benchmark file and submit a new PR
7. In a few days, if your benchmark is stable, bug Laith Sakka to enable running your benchmark on all PRs. If your a meta employee, you can find the dashboard here: internalfb.com/intern/unidash/dashboard/pt2_diff_time_metrics
