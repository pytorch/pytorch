If you just want to re-generate existing heuristics with already collected data for mixed_mm for A100/H100, run the following scripts:

`bash get_mixedmm_dataset.sh # Downloads A100 and H100 datasets`
`bash gen_mixedmm_heuristic_a100.sh # Generates A100 heuristic`
`bash gen_mixedmm_heuristic_h100.sh # Generates H100 heuristic`

If you want to collect new data, or generate a heuristic for another GPU, use the `generate_heuristic.sh` script:
First, go into the generate_heuristic.sh and modify the variables according to the comments.
Then run the script to perform benchmarks and collect training data:

`bash generate_heuristic.sh collect`

Depending on how many GPUs you are using, this might take a day.
Afterwards, run the script in order to learn the heuristic:

`bash generate_heuristic.sh generate`
