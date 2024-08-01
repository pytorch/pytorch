# mypy: ignore-errors
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_decision import AHTrainDecisionTree

import torch
from torch._inductor.autoheuristic.autoheuristic_utils import flex_attention_operations
from torch._inductor.kernel.flex_attention import get_default_config_fwd


class AHTrainDecisionTreeFlexAttention(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_base_arguments(self):
        super().add_base_arguments()
        self.parser.add_argument(
            "--gpu",
            type=str,
            help="Type of GPU to learn heuristic for.",
        )


    def add_new_features(self, results):
        ops = flex_attention_operations()
        added_categorical_features = []
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
            if op.is_categorical:
                added_categorical_features.append(op.name)
        return (results, added_categorical_features)

    def filter_df(self, df):
        # remove this config because there are large discrepancies between autotuning results and real performance
        df = df[df["choice"] != self.config_to_config_name((64, 16, 4, 3))]
        return df

    def get_dtype(self, row):
        if row["dtype_torch.float32"]:
            return torch.float32
        elif row["dtype_torch.float16"]:
            return torch.float16
        elif row["dtype_torch.bfloat16"]:
            return torch.bfloat16
        raise ValueError("dtype not supported")

    def config_to_config_name(self, config):
        return f"type=triton_BLOCK-M={config[0]}_BLOCK-K=-1_BLOCK-N={config[1]}_numstages={config[3]}_numwarps={config[2]}"

    def get_default_config(self, row):
        if self.args.gpu == "A100":
            capability = (8, 0)
        elif self.args.gpu == "H100":
            capability = (9, 0)
        else:
            raise ValueError(f"gpu {self.args.gpu} not supported")
        default_config = get_default_config_fwd(
            self.get_dtype(row), row["n"], capability
        )
        return self.config_to_config_name(default_config)

    def get_allowed_wrong_prediction_pct(self):
        return 1.0

    def get_test_and_val_size(self):
        return (0.01, 0.19)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        if predicted_config not in choice2time:
            # heuristic always returns "unsure" in such a case
            return False
        predicted_time = choice2time[predicted_config]

        # if any([choice2time[choice] * 1.10 < predicted_time for choice in choice2time.keys() if choice != predicted_config]):
        #     return True

        fallback_time = choice2time.get(self.get_default_config(row), None)
        if fallback_time is None:
            return False

        # we mark leaves as unsafe if there is a chance our choice will be 12% slower than fallback
        # we are okay with making the wrong choice, as long as our choice is better than fallback because
        # fallback is the default when max_autotune is false
        return 1.12 * fallback_time < predicted_time

    def get_grid_search_values(self):
        return {"max_depth": [6], "min_samples_leaf": [0.01], "criterion": ["entropy"]}


if __name__ == "__main__":
    train = AHTrainDecisionTreeFlexAttention()
    train.generate_heuristic()
