If you just want to re-generate existing heuristics with already collected data for mm for A100/H100, run the following scripts:

`bash get_mm_dataset.sh # Downloads A100 and H100 datasets`
`bash gen_heuristic_a100.sh # Generates A100 heuristic`
`bash gen_heuristic_h100.sh # Generates H100 heuristic`

If you want to collect new data, or generate a heuristic for another GPU, use the `generate_heuristic_mm.sh` script:
First, go into the generate_heuristic_mm.sh and modify the variables according to the comments. Then, run the script to perform benchmarks and collect training data:

`bash generate_heuristic.sh collect`

This will collect training data on random inputs. Depending on how many GPUs you are using, this might take a day.
If you use multiple GPU, you will have one file per GPU, e.g. "data_6.txt", "data_7.txt" if you used GPUs with id 6 and 7.
To merge this into a single file run:
`python torchgen/_autuoheuristic/merge_data.py mm_train.txt data_6.txt data_7.txt`

For mm, we also want to incorporate data from huggingface and TIMM models into the training data.

To collect data for huggingface, run the following command:

```
TORCHINDUCTOR_AUTOHEURISTIC_USE="" TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="mm" TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="hf_train_mm.txt" TORCHINDUCTOR_MAX_AUTOTUNE=1 time python ../../../benchmarks/dynamo/huggingface.py --ci --performance --timing --explain --inductor --device cuda --train --amp
```

To collect data for TIMM models, run the following command
```
TORCHINDUCTOR_AUTOHEURISTIC_USE="" TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="mm" TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="timm_train_mm.txt" TORCHINDUCTOR_MAX_AUTOTUNE=1 time python ../../../benchmarks/dynamo/timm_models.py --ci --performance --timing --explain --inductor --device cuda --train --amp
```

Afterwards, run the script in order to learn the heuristic:

`bash generate_heuristic_mm.sh generate`
