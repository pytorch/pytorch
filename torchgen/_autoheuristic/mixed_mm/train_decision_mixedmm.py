
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

    def get_allowed_wrong_prediction_pct(self):
        # it is okay to have wrong predictions
        # we introduce uncertainty by marking leaves as unsafe instead
        return 1.0

    def get_test_and_val_size(self):
        return (0.01, 0.19)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        if predicted_config not in choice2time:
            # heuristic always returns "unsure" in such a case
            return False
        predicted_time = choice2time[predicted_config]
        fallback_time = choice2time[self.get_default_config(row)]
        # we mark leaves as unsafe if there is a chance our choice will be 5% slower than fallback
        # we are okay with making the wrong choice, as long as our choice is better than fallback because
        # fallback is the default when max_autotune is false
        return 1.05 * fallback_time < predicted_time

    def get_grid_search_values(self):
        # A lot of different hyperparameters perform very similar on mixed_mm
        # it is kind of hard to automatically pick one so I just manually picked one with a small max_depth
        return {"max_depth": [5], "min_samples_leaf": [0.01], "criterion": ["entropy"]}


if __name__ == "__main__":
    train = AHTrainDecisionTreeMixedMM()
    train.generate_heuristic()
