# mypy: ignore-errors

import argparse
import sys
import warnings

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import gmean  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeRegressor  # type: ignore[import-untyped]

from torch._inductor.autoheuristic.autoheuristic import deserialize_data
from torch._inductor.autoheuristic.autoheuristic_utils import CHOICE_COL, FEEDBACK_COL


# TODO (AlnisM): Fix these warnings
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.",
)


class AHTrain:
    """
    This class is responsible for generating a heuristic by using data collected with AutoHeuristic. It will learn a
    regression tree that predicts a score that represents how well a specific choice will perform given an input.
    A higher score means a better choice. The heuristic will be generated in a file named <heuristic_name>.py in the
    torch/_inductor/autoheuristic/artifacts/ directory.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self):
        self.parser.add_argument(
            "dataset",
            type=str,
            help="Path to text file containing data collected with AutoHeuristic.",
        )
        self.parser.add_argument(
            "--nrows",
            type=int,
            default=None,
            help="Only read first n rows of the dataset.",
        )
        self.parser.add_argument(
            "--heuristic-name",
            type=str,
            default="learned_heuristic",
            help="Name of the heuristic to be generated.",
        )
        self.parser.add_argument(
            "--data",
            nargs=2,
            action="append",
            metavar=("TYPE", "PATH"),
            help="Specify name of datasets and file paths to be evaluated.",
        )

    def parse_args(self):
        return self.parser.parse_args()

    def generate_heuristic(self):
        self.args = self.parse_args()
        self.main(
            self.args.dataset, self.args.data, self.args.nrows, self.args.heuristic_name
        )

    def main(self, log_path, other_datasets, nrows, heuristic_name):
        (df, choices, cat_feature2cats, dummy_col_2_col_val, metadata) = self.get_df(
            log_path, nrows=nrows, apply_filters=True
        )
        df_train, df_val, df_test, feature_columns = self.custom_train_test_split(df)
        datasets = {"train": df_train, "val": df_val, "test": df_test}
        self.add_real_datasets(datasets, other_datasets, cat_feature2cats)

        # We will do a grid search over the values
        max_depths = [5, 10, 13, 15, 17, 20, 23, None]
        min_samples_leafs = [1, 2, 5, 10]
        choice_columns = [f"{CHOICE_COL}_{choice}" for choice in choices]
        (results_df, best_model, threshold) = self.train_and_evaluate_models(
            datasets, feature_columns, choice_columns, max_depths, min_samples_leafs
        )

        # prints results for all models and datasets
        print(results_df.to_string())

        # prints results grouped by dataset
        for set_name in results_df["dataset"].unique():
            dataset_results = results_df[results_df["dataset"] == set_name]
            dataset_results = dataset_results.sort_values(by="correct")
            print(dataset_results.to_string() + "\n")

        feature_names = feature_columns + choice_columns
        self.dt_to_python(
            best_model,
            metadata,
            feature_names,
            dummy_col_2_col_val,
            heuristic_name,
            threshold,
        )

    def filter_df(self, df):
        return df

    def get_df(self, log_path, cat_feature2cats=None, nrows=None, apply_filters=False):
        (df, metadata) = deserialize_data(log_path)
        numerical_features = metadata["numerical_features"]
        categorical_features = metadata["categorical_features"]
        choices = df[CHOICE_COL].unique().tolist()
        features = numerical_features + categorical_features
        if nrows is not None:
            df = df.head(nrows)

        df = self.filter_df(df)

        feature_columns = features

        def process_data(
            df,
            feature_columns,
            apply_filters,
            min_count_measurements=3,
            max_relative_std=5,
        ):
            # Calculate statistics for each input and choice combination
            def calculate_stats(group):
                count = len(group)
                mean = group[FEEDBACK_COL].mean()
                std = group[FEEDBACK_COL].std()
                relative_std = (std / mean) * 100 if mean != 0 else np.inf
                median = group[FEEDBACK_COL].median()
                return pd.Series(
                    {
                        "count": count,
                        "median_execution_time": median,
                        "relative_std": relative_std,
                    }
                )

            stats = (
                df.groupby(feature_columns + [CHOICE_COL])
                .apply(calculate_stats)
                .reset_index()
            )

            if apply_filters:
                # Remove unstables measurements
                valid_stats = stats[
                    (stats["count"] >= min_count_measurements)
                    & (stats["relative_std"] <= max_relative_std)
                ]
                # Keep only inputs with at least two valid choices
                valid_inputs = valid_stats.groupby(feature_columns).filter(
                    lambda x: len(x) >= 2
                )
            else:
                valid_inputs = stats

            # Compute the winner and ratios for each input
            def get_winner_and_speedups(group):
                mean_time = group["median_execution_time"].mean()
                winner = group.loc[group["median_execution_time"].idxmin(), CHOICE_COL]
                min_time = group["median_execution_time"].min()
                max_time = group["median_execution_time"].max()

                group["winner"] = winner
                group["speedup"] = max_time / min_time
                group["target"] = mean_time / group["median_execution_time"]

                return group[
                    feature_columns + [CHOICE_COL, "winner", "speedup", "target"]
                ]

            results = (
                valid_inputs.groupby(feature_columns)
                .apply(get_winner_and_speedups)
                .reset_index(drop=True)
            )

            return results

        results = process_data(df, feature_columns, apply_filters)
        (results, added_categorical_features) = self.add_new_features(results)
        categorical_features += added_categorical_features
        categorical_features += [CHOICE_COL]

        # Doing this here because if we create another df for testing purposes
        # and that other df does not contain all categories for a categorical feature,
        # pd.dummies will not create columns for the missing categories
        if not cat_feature2cats:
            cat_feature2cats = {}
        for cat_feature in categorical_features:
            if cat_feature in cat_feature2cats:
                categories = cat_feature2cats[cat_feature]
            else:
                categories = results[cat_feature].unique()
                cat_feature2cats[cat_feature] = categories
            results[cat_feature] = pd.Categorical(
                results[cat_feature], categories=categories
            )

        dummy_col_2_col_val = {}
        for col in categorical_features:
            unique_vals = results[col].unique()
            for val in unique_vals:
                dummy_col_2_col_val[f"{col}_{val}"] = (col, val)

        # one-hot encode categorical features
        results = pd.get_dummies(results, columns=categorical_features)
        return (results, choices, cat_feature2cats, dummy_col_2_col_val, metadata)

    def custom_train_test_split(
        self, df, test_size=0.2, val_size=0.25, random_state=42
    ):
        # We want to make sure that rows with the same input but different choice are kept in the same set
        exclude_columns = ["speedup", "winner", "target"]
        feature_columns = [
            col
            for col in df.columns
            if col not in exclude_columns and not col.startswith(CHOICE_COL + "_")
        ]
        df["input_id"] = df.groupby(feature_columns).ngroup()

        # Get unique input IDs
        unique_inputs = df["input_id"].unique()

        # Split unique inputs into train+val and test
        train_val_inputs, test_inputs = train_test_split(
            unique_inputs, test_size=test_size, random_state=random_state
        )

        # Split train+val inputs into train and val
        train_inputs, val_inputs = train_test_split(
            train_val_inputs, test_size=val_size, random_state=random_state
        )

        # Create masks for each set
        train_mask = df["input_id"].isin(train_inputs)
        val_mask = df["input_id"].isin(val_inputs)
        test_mask = df["input_id"].isin(test_inputs)

        # Split the dataframe
        df_train = df[train_mask]
        df_val = df[val_mask]
        df_test = df[test_mask]

        # Remove the temporary input_id column
        df_train = df_train.drop("input_id", axis=1)
        df_val = df_val.drop("input_id", axis=1)
        df_test = df_test.drop("input_id", axis=1)

        return df_train, df_val, df_test, feature_columns

    def train_and_evaluate_models(
        self,
        datasets,
        feature_columns,
        choice_columns,
        max_depths,
        min_samples_leafs,
        threshold=0.99,
    ):
        results = []
        df_train = datasets["train"]
        df_val = datasets["val"]

        best_model = None
        best_model_threshold = 0
        max_correct_predictions = -1
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                print(
                    f"Evaluating max_depth={max_depth}, min_samples_leaf={min_samples_leaf}"
                )
                model = DecisionTreeRegressor(
                    random_state=42,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                )
                model.fit(
                    df_train[feature_columns + choice_columns], df_train["target"]
                )

                # we first compute a safe threshold: this threshold ensures that on the validation set,
                # if the heuristic returns a choice, the choice will be correct, although a high threshold
                # can lead to a lot of 'unsure' choices
                eval_result = self.evaluate_model(
                    model, df_val, feature_columns, choice_columns, threshold
                )
                safe_threshold = eval_result["wrong_max_ratio"]
                for dataset_name, dataset in datasets.items():
                    eval_result = self.evaluate_model(
                        model, dataset, feature_columns, choice_columns, safe_threshold
                    )
                    print(eval_result)
                    if dataset_name == "val":
                        eval_correct = eval_result["correct"]
                        if eval_correct > max_correct_predictions:
                            best_model = model
                            best_model_threshold = safe_threshold
                            max_correct_predictions = eval_correct
                    results.append(
                        {
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "dataset": dataset_name,
                            "correct": eval_result["correct"],
                            "wrong": eval_result["wrong"],
                            "unsure": eval_result["unsure"],
                            "total": eval_result["total"],
                            "max_wrong_speedup": eval_result["max_wrong_speedup"],
                            "gman_wrong_speedup": eval_result["gman_wrong_speedup"],
                            "threshold": safe_threshold,
                        }
                    )

        return (pd.DataFrame(results), best_model, best_model_threshold)

    def evaluate_model(self, model, df, feature_columns, choice_columns, threshold):
        def predict_winner(group):
            predictions = model.predict(group[feature_columns + choice_columns])

            # Find the index of the maximum prediction (best choice)
            best_choice_index = np.argmax(predictions)

            # Get the corresponding choice
            predicted_choice = (
                group[choice_columns].iloc[best_choice_index].idxmax().split("_")[-1]
            )

            # Calculate the ratio between the best and second-best prediction
            sorted_predictions = np.sort(predictions)[::-1]
            top_pred_ratio = (
                sorted_predictions[0] / sorted_predictions[1]
                if len(sorted_predictions) > 1
                else np.inf
            )

            # If the best choice is not "significantly" better than the second best choice,
            # the learned heuristic will return "unsure"
            if top_pred_ratio <= threshold:
                predicted_winner = "unsure"
            else:
                predicted_winner = predicted_choice

            actual_winner = group["winner"].iloc[0]
            is_correct = (
                predicted_winner == actual_winner
                if predicted_winner != "unsure"
                else "unsure"
            )

            return pd.Series(
                {
                    "predicted_winner": predicted_winner,
                    "ratio": top_pred_ratio,
                    "actual_winner": actual_winner,
                    "is_correct": is_correct,
                    "speedup": group["speedup"].iloc[
                        0
                    ],  # Speedup is the same for all rows in the group
                }
            )

        results = df.groupby(feature_columns).apply(predict_winner).reset_index()
        correct = (results["is_correct"].eq(True)).sum()
        unsure = (results["is_correct"] == "unsure").sum()
        wrong_results = results[results["is_correct"].eq(False)]
        wrong = len(wrong_results)

        # Calculate max and geometric mean of speedup for wrong predictions
        # Used for debugging purposes
        wrong_speedups = wrong_results["speedup"]
        max_wrong_speedup = wrong_speedups.max() if not wrong_speedups.empty else np.nan
        geo_mean_wrong_speedup = (
            gmean(wrong_speedups) if not wrong_speedups.empty else np.nan
        )
        wrong_max_ratio = wrong_results["ratio"].max()

        total = correct + wrong + unsure
        return {
            "correct": correct,
            "wrong": wrong,
            "unsure": unsure,
            "total": total,
            "max_wrong_speedup": max_wrong_speedup,
            "gman_wrong_speedup": geo_mean_wrong_speedup,
            "wrong_max_ratio": wrong_max_ratio,
        }

    def add_new_features(self, results):
        return (results, [])

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa
    ):
        boiler_plate = f"""# flake8: noqa: B950

from torch._inductor.autoheuristic.autoheuristic_utils import AHContext, AHMetadata, Choice, CHOICE_COL
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristic,
)

class {heuristic_name}(LearnedHeuristic):

    def __init__(self) -> None:
        pass

    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == {shared_memory}
            and str(metadata.device_capa) == "{device_capa}"
        )

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        context.context_dict[CHOICE_COL] = choice
        return self.predict(context)

    def get_speedup_threshold(self) -> float:
        return {threshold}

    def get_name(self) -> str:
        return '{opt_name}'"""
        return boiler_plate

    def dt_to_python(
        self,
        dt,
        metadata,
        feature_names,
        dummy_col_2_col_val,
        heuristic_name,
        threshold,
    ):
        tree_ = dt.tree_
        feature_name = [
            feature_names[i] if i != -1 else "undefined!" for i in tree_.feature
        ]

        lines = []
        device_capa = metadata["device_capa"]
        device_capa_str = f"({device_capa[0]}, {device_capa[1]})"
        opt_name = metadata["name"]
        lines.append(
            self.codegen_boilerplate(
                heuristic_name,
                opt_name,
                threshold,
                metadata["shared_memory"],
                device_capa_str,
            )
        )
        fn_def = "\n    def predict(self, context: AHContext) -> float:"
        lines.append(fn_def)

        def dt_to_python(node, depth):
            indent = "    " * (depth + 1)
            false_predicate = ""
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                if name in dummy_col_2_col_val:
                    (orig_name, value) = dummy_col_2_col_val[name]
                    predicate = f"{indent}if str(context.get_value('{orig_name}')) != '{value}':"
                    if threshold != 0.5:
                        print(f"expected threshold to be 0.5 but is {threshold}")
                        sys.exit(1)
                else:
                    predicate = (
                        f"{indent}if context.get_value('{name}') <= {threshold}:"
                    )
                lines.append(predicate)
                dt_to_python(tree_.children_left[node], depth + 1)
                lines.append(f"{indent}else:")
                dt_to_python(tree_.children_right[node], depth + 1)
            else:
                value = tree_.value[node][0][0]
                lines.append(f"{indent}return {str(value)}")

        dt_to_python(0, 1)

        output_file = (
            f"../../torch/_inductor/autoheuristic/artifacts/_{heuristic_name}.py"
        )
        path = f"{output_file}"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def add_real_datasets(self, datasets, other_datasets, cat_feature2cats):
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path, cat_feature2cats=cat_feature2cats, apply_filters=False
                )
                datasets[name] = df_other


if __name__ == "__main__":
    train = AHTrain()
    train.generate_heuristic()
