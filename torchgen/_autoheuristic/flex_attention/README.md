If you just want to re-generate existing heuristics with already collect data for flex_attention for A100/H100, run the following scripts:

bash get_flexattention_datasets.sh # Downloads A100 and H100 datasets
bash gen_flexattention_heuristic_a100.sh # Generates A100 heuristic
bash gen_flexattention_heuristic_h100.sh # Generates H100 heuristic
