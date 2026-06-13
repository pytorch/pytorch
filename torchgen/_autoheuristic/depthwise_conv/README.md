## Regenerating the current heuristic
To regenerate the current heuristic with the original data, run the following scripts:

```
bash get_depthwiseconv_dataset.sh

python train_decision_depthwiseconv.py *.csv
```

## Benchmarking
To collect new data, run the benchmarking script:

`python gen_data_depthwiseconv.py [--device DEVICE]`

The `--device` flag is optional, and allows you to specify a custom (shorter) label for the GPU being tested.
Output will be saved to `data_depthwise_conv_[DEVICE].csv`. Depending on the GPU, this will likely take 30 minutes to 2 hours.

## Heuristic generation
To generate a new heuristic from benchmarking data, run the training script:

`python train_decision_depthwiseconv.py [input files ...]`

At least one input file must be provided. If multiple files are provided, the training script will first combine all inputs into
one dataset, averaging the normalized speedup results.

There are several options that can be provided to the training script, with `--tolerance` being the most important. This option
excludes data from decision tree training if the normalized speedup value is within tolerance, which can make a large impact on
final tree complexity and accuracy. For example, consider the following benchmarking results with `--tolerance 0.1`:

|id|cudnn_speedup_all|
|---|---|
|A|1.4|
|B|0.95|
|C|1.05|
|D|0.8|

Samples A and D will be included in training, because they are outside the tolerance window of `0.1`:
```
abs(1.4 - 1) > 0.1
abs(0.8 - 1) > 0.1
```
Samples B and C will be excluded from training, because they are within the tolerance window of `0.1`.

### Full options information:
```
usage: train_decision_depthwiseconv.py [-h] [--tolerance TOLERANCE] [--max-depth MAX_DEPTH] [--max-leaf-nodes MAX_LEAF_NODES]
                                       [--min-samples-split MIN_SAMPLES_SPLIT] [--min-samples-leaf MIN_SAMPLES_LEAF]
                                       [--criterion {gini,entropy,log_loss}] [--seed SEED]
                                       input_files [input_files ...]

positional arguments:
  input_files           Paths to processed CSV files

options:
  -h, --help            show this help message and exit
  --tolerance TOLERANCE
                        Tolerance threshold (default: 0.0)
  --max-depth MAX_DEPTH
                        Maximum tree depth (default: None = unlimited)
  --max-leaf-nodes MAX_LEAF_NODES
                        Maximum number of leaf nodes (default: None = unlimited)
  --min-samples-split MIN_SAMPLES_SPLIT
                        Minimum samples to split a node (default: 2)
  --min-samples-leaf MIN_SAMPLES_LEAF
                        Minimum samples in leaf node (default: 1)
  --criterion {gini,entropy,log_loss}
                        Split criterion (default: gini)
  --seed SEED           Random seed (default: 42)
```