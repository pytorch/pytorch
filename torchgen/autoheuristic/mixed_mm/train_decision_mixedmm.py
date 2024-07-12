# mypy: ignore-errors
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_decision import AHTrainDecisionTree

from torch._inductor.autoheuristic.autoheuristic_utils import mixed_mm_operations


class AHTrainDecisionTreeMixedMM(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        ops = mixed_mm_operations()
        added_categorical_features = []
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
            if op.is_categorical:
                added_categorical_features.append(op.name)
        return (results, added_categorical_features)

    def get_default_config(self, row):
        return "extern_fallback_mixed_mm"


if __name__ == "__main__":
    train = AHTrainDecisionTreeMixedMM()
    train.generate_heuristic()
