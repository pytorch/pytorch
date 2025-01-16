# mypy: ignore-errors
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_regression import AHTrainRegressionTree

from torch._inductor.fx_passes.pad_mm import pad_mm_operations


class AHTrainPadMM(AHTrainRegressionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        ops = pad_mm_operations()
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
        added_categorical_features = [op.name for op in ops if op.is_categorical]
        return (results, added_categorical_features)


if __name__ == "__main__":
    train = AHTrainPadMM()
    train.generate_heuristic()
