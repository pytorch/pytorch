If you just want to re-generate existing heuristics with already collect data for mixed_mm for A100/H100, run the following scripts:

`bash get_mm_dataset.sh # Downloads A100 and H100 datasets`
`bash get_mixedmm_heuristic_a100.sh # Generates A100 heuristic`
`bash get_mixedmm_heuristic_h100.sh # Generates H100 heuristic`

If you want to collect new data, or generate a heuristic for another GPU, use the `generate_heuristic_mm.sh` script:
First, go into the generate_heuristic_mm.sh and modify the variables according to the comments. Then, run the script to perform benchmarks and collect training data:

`bash generate_heuristic.sh collect`

This will collect training data on random inputs. Depending on how many GPUs you are using, this might take a day.

For mm, we also want to incorporate data from huggingface and TIMM models.

To collect data for huggingface, run the following command:

```
TORCHINDUCTOR_AUTOHEURISTIC_USE="" TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="mm" TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="a100/hf_train_mm.txt" TORCHINDUCTOR_MAX_AUTOTUNE=1 time python ../../../benchmarks/dynamo/huggingface.py --ci --performance --timing --explain --inductor --device cuda --train --amp
```

To collect data for TIMM models, run the following command
```
TORCHINDUCTOR_AUTOHEURISTIC_USE="" TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="mm" TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="a100/timm_train_mm.txt" TORCHINDUCTOR_MAX_AUTOTUNE=1 time python ../../../benchmarks/dynamo/timm_models.py --ci --performance --timing --explain --inductor --device cuda --train --amp
```

Afterwards, run the script in order to learn the heuristic:

`bash generate_heuristic.sh generate`
