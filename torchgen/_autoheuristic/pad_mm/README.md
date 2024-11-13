If you just want to re-generate existing heuristics with already collected data for pad_mm for A100, run the following scripts:

`bash get_padmm_dataset.sh # Downloads A100`
`bash gen_pad_mm_a100.sh # Generates A100 heuristic`

If you want to collect new data, or generate a heuristic for another GPU, use the `generate_heuristic_pad_mm.sh` script:
First, go into the generate_heuristic_mm.sh and modify the variables according to the comments. Then, run the script to perform benchmarks and collect training data:

`bash generate_heuristic_pad_mm.sh collect`

This will collect training data on random inputs. Depending on how many GPUs you are using, this might take a day.
Afterwards, run the script in order to learn the heuristic:

`bash generate_heuristic_pad_mm.sh generate`
