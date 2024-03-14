set -ex
rm -f /tmp/torchinductor_$USER -rf

# MODEL=MobileBertForQuestionAnswering # 1m35.617s (8.942s) -> 1m27.277s (3.034s)
MODEL=DebertaForQuestionAnswering # 1m5.640s (2.353s+0.097s) -> 1m5.447s (2.324s+0.085s)


# export TORCH_COMPILE_CPROFILE=1
BIN=python
# BIN="kernprof -l"
# BIN="strobeclient run --profiler pyperf"
# BIN=/home/aorenste/fbcode/f3/tools/perf_test/pyperf.sh
# BIN="pyinstrument -r html -o foo.html -i 0.0005"
time $BIN benchmarks/dynamo/huggingface.py --training --amp --performance --only $MODEL --backend=inductor
# python -m line_profiler huggingface.py.lprof
# /home/aorenste/fbcode/f3/tools/perf_test/pyperf.sh "python benchmarks/dynamo/huggingface.py --training --amp --performance --only $MODEL --backend=inductor"
