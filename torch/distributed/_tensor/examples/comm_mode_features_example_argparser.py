import argparse

parser = argparse.ArgumentParser(
    description="comm_mode_feature examples",
    formatter_class=argparse.RawTextHelpFormatter,
)
example_prompt = (
    "choose one comm_mode_feature example from below:\n"
    "\t1. MLP_distributed_sharding_display\n"
    "\t2. MLPStacked_distributed_sharding_display\n"
    "\t3. MLP_module_tracing\n"
    "\t4. transformer_module_tracing\n"
    "e.g. you want to try the MLPModule sharding display example, please input 'MLP_distributed_sharding_display'\n"
)
parser.add_argument("-e", "--example", help=example_prompt, required=True)
example = parser.parse_args().example


def args() -> str:
    return example
