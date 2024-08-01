# mypy: ignore-errors

import itertools
import json
import math
import warnings


warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from ah_tree import DecisionTree
from scipy.stats import gmean
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from train import AHTrain


class AHTrainDecisionTree(AHTrain):
    def __init__(self):
        super().__init__()

    def get_time(self, row, choice):
        choices_feedback = json.loads(row["choice2time"])
        return choices_feedback.get(choice, None)

    def debug_time(self, row, top_k_choices):
        choices_feedback = json.loads(row["choice2time"])
        timings = sorted(choices_feedback.items(), key=lambda x: x[1])
        for choice, time in timings:
            result = f"{choice} {time}"
            if choice in top_k_choices:
                result += " TOPK"
            print(result)

    def top_k_classes(self, model, probas, k, avail_choices):
        # Get classes and their corresponding probabilities
        classes = model.classes_
        class_proba_pairs = list(zip(classes, probas))

        # Sort by probability (descending) and filter out zero probabilities
        sorted_classes = [
            c
            for c, p in sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)
            if p > 0 and c in avail_choices
        ]

        # Return top k choices
        top_k_choices = sorted_classes[:k]
        top_k_choices += self.ranking_always_included_choices()
        top_k_choices = list(dict.fromkeys(top_k_choices))
        return top_k_choices

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        """
        Can be overridden by subclasses to define their own logic for deciding when a leaf is unsafe. Returns a sample
        that landed in the leaf, the choice predicted by the tree, and a dictionary that maps each choice to the
        execution time. One can for example decide to mark a leaf as unsafe if the predicted choice is 2x slower
        than the fastest choice.
        If a leaf is unsafe, the learned heuristic will always return 'unsure' if an input lands in that leaf.
        """

        return False

    def get_unsafe_leaves(self, model, df, feature_columns):
        """
        Given a trained decision tree, and a dataframe containing the training data, returns a list of unsafe leaves.
        """
        X = df[feature_columns]
        y = df["winner"]
        leaf_ids = model.apply(X)
        unique_leaves = np.unique(leaf_ids)

        unsafe_leaves = []
        # Iterate over each leaf
        for leaf in unique_leaves:
            leaf_mask = leaf_ids == leaf
            # Get samples that land in this leaf
            leaf_X = X[leaf_mask]

            predicted_config = model.predict(leaf_X.iloc[[0]])[0]

            # For each sample, check if we should mark the leaf as unsafe
            for idx, row in leaf_X.iterrows():
                choice2time = json.loads(df.loc[idx, "choice2time"])
                if self.is_unsafe_leaf(row, predicted_config, choice2time):
                    unsafe_leaves.append(leaf)
                    break
        return unsafe_leaves

    def get_allowed_wrong_prediction_pct(self):
        """
        This is used to determine a threshold for when a learned heuristic returns 'unsure'.
        If this function returns 0.01, we will set the probability required for the decision tree to return a decision
        such that at most 1% of the predictions will be wrong on the validation set.
        """
        return 0.01

    def get_grid_search_values(self):
        """
        Standard values for grid search. Can be overriden.
        """
        return {
            "max_depth": [5, 6, 7],
            "min_samples_leaf": [1, 5, 10, 0.01, 0.05, 0.02],
            "criterion": ["gini", "entropy"],
        }

    def predict(self, model, df, feature_columns):
        """
        Returns the predictions, probabilities, and leaf ids for a given dataframe.
        """
        predictions = model.predict(df[feature_columns])
        proba = model.predict_proba(df[feature_columns])
        leaf_ids = model.apply(df[feature_columns])
        return predictions, proba, leaf_ids

    def ranking_num_choices(self):
        # if the heuristic is used for ranking, this function returns the number
        # of choices that the heuristic will return
        return 5

    def train_and_evaluate_models(
        self,
        datasets,
        max_depths,
        min_samples_leafs,
        criterion_list,
        feature_columns,
        ranking=False,
    ):
        """
        Does a grid search over max_depths, min_samples_leafs, and criterion_list and returns the best model.
        """

        results = []
        best_model = None
        best_model_safe_proba = 0
        best_model_num_correct = 0
        best_model_num_wrong = 0
        best_model_unsafe_leaves = []
        for max_depth, min_samples_leaf, criterion in itertools.product(
            max_depths, min_samples_leafs, criterion_list
        ):
            print(
                f"max_depth={max_depth} min_samples_leaf={min_samples_leaf} criterion={criterion}"
            )
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
            )
            df_train = datasets["train"]
            df_val = datasets["val"]
            if ranking:
                model.fit(
                    df_train[feature_columns],
                    df_train["winner"],
                    sample_weight=df_train["relative_performance"],
                )
            else:
                model.fit(df_train[feature_columns], df_train["winner"])

            model = DecisionTree(model, feature_columns)

            if ranking:
                model.prune(df_train, "winner", k=self.ranking_num_choices())

            unsafe_leaves = self.get_unsafe_leaves(model, df_train, feature_columns)
            predictions, proba, leaf_ids = self.predict(model, df_val, feature_columns)

            wrong_pct = self.get_allowed_wrong_prediction_pct()
            safe_proba = self.get_results(
                model,
                predictions,
                df_val,
                probas=proba,
                return_safe_proba=True,
                wrong_pct=wrong_pct,
                unsafe_leaves=unsafe_leaves,
                leaf_ids=leaf_ids,
                k=self.ranking_num_choices(),
                ranking=ranking,
            )
            print(f"safe_proba={safe_proba}")

            def eval(name, df):
                if ranking:
                    # when ranking is enabled, we duplicate each input for each choice that
                    # is almost as good as the best choice
                    # we do not want to evaluate the same input multiple times, so we remove duplicates here
                    df = df[df["winner"] == df["actual_winner"]]
                predictions, proba, leaf_ids = self.predict(model, df, feature_columns)
                eval_result = self.get_results(
                    model,
                    predictions,
                    df,
                    probas=proba,
                    threshold=safe_proba,
                    unsafe_leaves=unsafe_leaves,
                    leaf_ids=leaf_ids,
                    k=self.ranking_num_choices(),
                    ranking=ranking,
                )
                if name == "val":
                    nonlocal best_model_num_correct
                    nonlocal best_model_num_wrong
                    nonlocal best_model_safe_proba
                    nonlocal best_model
                    nonlocal best_model_unsafe_leaves
                    num_correct = eval_result["correct"]
                    num_wrong = eval_result["wrong"]
                    num_total = eval_result["total"]
                    if num_wrong <= num_total * wrong_pct:
                        if num_correct > best_model_num_correct:
                            print(
                                f"new best model with {num_correct} correct and {num_wrong} wrong"
                            )
                            best_model = model
                            best_model_num_correct = num_correct
                            best_model_num_wrong = num_wrong
                            best_model_safe_proba = safe_proba
                            best_model_unsafe_leaves = unsafe_leaves

                results.append(
                    (
                        name,
                        criterion,
                        max_depth,
                        min_samples_leaf,
                        eval_result["correct"],
                        eval_result["wrong"],
                        eval_result["unsure"],
                        eval_result["total"],
                        eval_result["wrong_max_speedup"],
                        eval_result["wrong_gmean_speedup"],
                        eval_result["top_k_correct"],
                        eval_result["top_k_wrong"],
                        eval_result["wrong_max_speedup_k"],
                        eval_result["wrong_gmean_speedup_k"],
                        eval_result["top_k_unsure"],
                        eval_result["max_speedup_default"],
                        eval_result["gmean_speedup_default"],
                        eval_result["max_slowdown_default"],
                        eval_result["non_default_predictions"],
                        eval_result["default_better"],
                    )
                )

            for dataset_name, dataset in datasets.items():
                eval(dataset_name, dataset)

        return (
            pd.DataFrame(
                results,
                columns=[
                    "set",
                    "crit",
                    "max_depth",
                    "min_samples_leaf",
                    "correct",
                    "wrong",
                    "unsure",
                    "total",
                    "wrong_max_spdup",
                    "wrong_gman_spdup",
                    "top_k_correct",
                    "top_k_wrong",
                    "wrong_max_spdup_k",
                    "wrong_gman_spdup_k",
                    "top_k_unsure",
                    "max_spdup_default",
                    "gman_spdup_default",
                    "max_slowdown_default",
                    "non_default_preds",
                    "default_better",
                ],
            ),
            best_model,
            best_model_safe_proba,
            best_model_unsafe_leaves,
        )

    def get_test_and_val_size(self):
        """
        Returns the size of the test and validation sets.
        """
        return (0.15, 0.15)

    def prepare_datasets(self, df, other_datasets, cat_feature2cats, ranking=False):
        """
        Splits the dataframe into train, val, and test sets.
        Also adds other datasets, specified by the user, to the train set.
        """
        test_size, val_size = self.get_test_and_val_size()
        # Split into train+val and test
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=42
        )

        # Split train+val inputs into train and val
        train_val_size = 1 - test_size
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_size / train_val_size, random_state=42
        )
        datasets = {"train": df_train, "val": df_val, "test": df_test}
        self.add_real_datasets(datasets, other_datasets, cat_feature2cats, ranking)
        return datasets

    def export_to_dot(self, best_model, df, feature_columns):
        """
        Export a learned decision tree to a dot file.
        """
        dot_str = best_model.to_dot()
        with open("best_model.dot", "w") as f:
            f.write(dot_str)

    def get_feature_columns(self, df):
        """
        The dataframe contains columns that are not features, such as 'winner', 'speedup' that are only used for
        debugging purposes. This function returns the columns that are actually features.
        """
        exclude_columns = [
            "speedup",
            "winner",
            "target",
            "avail_choices",
            "choice2time",
            "index",
            "actual_winner",
            "relative_performance",
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns

    def add_training_data(self, df_train, datasets):
        return datasets["train"]

    def main(
        self,
        log_path,
        other_datasets,
        nrows,
        heuristic_name,
        save_dot=False,
        ranking=False,
    ):
        """
        Main function that trains a decision tree and generates a heuristic.
        """
        # TODO: Enable apply_filters
        (df, choices, cat_feature2cats, dummy_col_2_col_val, metadata) = self.get_df(
            log_path, nrows=nrows, apply_filters=False, add_near_best=ranking
        )
        self.dummy_col_2_col_val = dummy_col_2_col_val
        datasets = self.prepare_datasets(df, other_datasets, cat_feature2cats, ranking)
        df_train = self.add_training_data(datasets["train"], datasets)
        datasets["train"] = df_train
        print(datasets["train"]["winner"].value_counts().to_string())

        feature_columns = self.get_feature_columns(df)
        grid_search_values = self.get_grid_search_values()
        max_depths = grid_search_values["max_depth"]
        min_samples_leafs = grid_search_values["min_samples_leaf"]
        criterion_list = grid_search_values["criterion"]
        (
            results_df,
            best_model,
            best_model_safe_proba,
            unsafe_leaves,
        ) = self.train_and_evaluate_models(
            datasets,
            max_depths,
            min_samples_leafs,
            criterion_list,
            feature_columns,
            ranking=ranking,
        )

        if ranking:
            columns_to_keep = [
                "set",
                "crit",
                "max_depth",
                "min_samples_leaf",
                "total",
                "top_k_correct",
                "top_k_wrong",
                "top_k_unsure",
                "wrong_max_spdup_k",
                "wrong_gman_spdup_k",
            ]
            results_df = results_df[columns_to_keep]
        # prints results for all models and datasets
        print(results_df.to_string())

        sort_metric = "top_k_correct" if ranking else "correct"
        # prints results grouped by dataset
        for set_name in results_df["set"].unique():
            dataset_results = results_df[results_df["set"] == set_name]
            dataset_results = dataset_results.sort_values(by=sort_metric)
            print(dataset_results.to_string() + "\n")

        if best_model is not None:
            if save_dot:
                self.export_to_dot(best_model, df, feature_columns)
            self.codegen(
                best_model,
                metadata,
                heuristic_name,
                best_model_safe_proba,
                dummy_col_2_col_val,
                unsafe_leaves,
            )
        else:
            print(
                "All learned models have too many wrong predictions, so no heuristic was generated"
            )

    def get_df(
        self,
        log_path,
        cat_feature2cats=None,
        nrows=None,
        apply_filters=False,
        add_near_best=False,
    ):
        """
        Parses the log file and processes the data into a dataframe that can be used for training.
        """
        (df, metadata, features, categorical_features, choices) = self.parse_log(
            log_path, nrows
        )

        def calculate_stats(group):
            count = len(group)
            has_inf = np.isinf(group["feedback"]).any()
            if has_inf:
                relative_std = np.inf
                median = np.inf
            else:
                mean = group["feedback"].mean()
                std = group["feedback"].std()
                relative_std = (std / mean) * 100 if mean != 0 else np.inf
                median = group["feedback"].median()
            return pd.Series(
                {
                    "count": count,
                    "relative_std": relative_std,
                    "median_execution_time": median,
                }
            )

        feature_columns = features
        stats = (
            df.groupby(feature_columns + ["choice"], as_index=False)
            .apply(calculate_stats, include_groups=False)
            .reset_index()
        )

        # TODO: We have to be careful with removing certain choices, because if we e.g. remove the winner, the
        # heuristic will end up learning wrong things. But, execution times with high variance are also bad
        if apply_filters:
            # Filter out inputs with less than 3 measurements or high relative std
            valid_stats = stats[(stats["count"] >= 3) & (stats["relative_std"] <= 5)]
            # Group by input features and count how many valid choices we have for each input
            valid_inputs = valid_stats.groupby(feature_columns).filter(
                lambda x: len(x) >= 2
            )
        else:
            valid_inputs = stats

        # Compute the winner and speedup for each valid input
        def get_winner_and_speedup(group):
            assert len(group) >= 2, "Need at least 2 choices"

            sorted_group = group.sort_values("median_execution_time")
            winner = sorted_group.iloc[0]["choice"]
            winning_time = sorted_group.iloc[0]["median_execution_time"]
            second_best_time = sorted_group.iloc[1]["median_execution_time"]
            speedup = second_best_time / winning_time
            unique_choices = group["choice"].unique()

            choice2time = {}
            for row in group.itertuples():
                choice2time[row.choice] = row.median_execution_time

            assert len(unique_choices) == len(
                group
            ), f"len(unique_choices) != len(group): {len(unique_choices)} != {len(group)}"

            return pd.Series(
                {
                    "winner": winner,
                    "speedup": speedup,
                    "avail_choices": unique_choices,
                    "choice2time": json.dumps(choice2time),
                }
            )

        results = (
            valid_inputs.groupby(feature_columns, as_index=False)
            .filter(lambda x: len(x) >= 2)
            .groupby(feature_columns, as_index=False)
            .apply(get_winner_and_speedup, include_groups=False)
            .reset_index()
        )

        def add_near_best_configs(df):
            new_rows = []

            for index, row in df.iterrows():
                dictionary = json.loads(row["choice2time"])
                min_value = min(dictionary.values())

                for key, value in dictionary.items():
                    new_row = row.copy()
                    relative_performance = min_value / value
                    new_row["relative_performance"] = relative_performance
                    if relative_performance is None or relative_performance is np.inf:
                        breakpoint()
                    new_row["actual_winner"] = row["winner"]
                    new_row["winner"] = key
                    if relative_performance >= 0.98:
                        new_rows.append(new_row)

            return pd.DataFrame(new_rows).reset_index(drop=True)

        if add_near_best:
            results = add_near_best_configs(results)
        (results, added_categorical_features) = self.add_new_features(results)
        categorical_features += added_categorical_features

        (
            results,
            cat_feature2cats,
            dummy_col_2_col_val,
        ) = self.handle_categorical_features(
            cat_feature2cats, categorical_features, results
        )
        return (results, choices, cat_feature2cats, dummy_col_2_col_val, metadata)

    def ranking_always_included_choices(self):
        return []

    def get_results(
        self,
        model,
        predictions,
        df,
        probas,
        return_safe_proba=False,
        wrong_pct=0.01,
        threshold=0.0,
        k=10,
        unsafe_leaves=None,
        leaf_ids=None,
        ranking=False,
    ):
        """
        Custom evaluation function that evaluates a learned decision tree.
        """

        def compute_speedup_over_default(default_config, pred, df, i, predicted_time):
            nonlocal num_non_default_predictions
            nonlocal speedups_over_default
            nonlocal num_default_better
            if default_config is not None:
                if pred != default_config:
                    num_non_default_predictions += 1
                default_time = self.get_time(df.iloc[i], default_config)
                # TODO: We should keep track of how often this happens
                if default_time is not None and not math.isinf(default_time):
                    speedup_over_default = default_time / predicted_time
                    if speedup_over_default < 1:
                        num_default_better += 1
                    speedups_over_default.append(speedup_over_default)

        y_true = df["actual_winner"] if ranking else df["winner"]
        num_correct = 0
        num_wrong = 0
        num_unsure = 0
        wrong_probas = []
        i = 0
        speedups_wrong = []
        num_correct_top_k = 0
        wrong_speedups_top_k = []
        top_k_unsure = 0
        speedups_over_default = []
        num_non_default_predictions = 0
        num_default_better = 0
        num_wrong_top_k = 0
        for pred, true, prob, leaf_id in zip(predictions, y_true, probas, leaf_ids):
            avail_choices = df["avail_choices"].iloc[i]
            top_k_choices = self.top_k_classes(
                model, probas[i], k=k, avail_choices=avail_choices
            )
            predicted_time = self.get_time(df.iloc[i], pred)
            assert true in avail_choices, f"{true} not in {avail_choices}"

            default_config = self.get_default_config(df.iloc[i])

            max_prob = max(prob)
            if (
                leaf_id in unsafe_leaves
                or pred not in avail_choices
                or (max_prob != 1.0 and max(prob) <= threshold)
            ):
                num_unsure += 1
                speedups_over_default.append(1.0)
            elif pred == true:
                compute_speedup_over_default(
                    default_config, pred, df, i, predicted_time
                )
                num_correct += 1
            else:
                compute_speedup_over_default(
                    default_config, pred, df, i, predicted_time
                )
                num_wrong += 1
                wrong_probas.append(max_prob)
                best_time = self.get_time(df.iloc[i], true)
                wrong_speedup = predicted_time / best_time
                speedups_wrong.append(wrong_speedup)

            if true in top_k_choices:
                num_correct_top_k += 1
            else:
                times = []
                for choice in top_k_choices:
                    time = self.get_time(df.iloc[i], choice)
                    if time is not None:
                        times.append(time)
                best_time = self.get_time(df.iloc[i], true)
                min_time = min(times) if times else None
                if min_time is not None:
                    speedup = min_time / best_time
                    wrong_speedups_top_k.append(speedup)
                    num_wrong_top_k += 1
                else:
                    top_k_unsure += 1
            i += 1

        if return_safe_proba:
            wrong_probas.sort()
            total = len(predictions)
            num_wrong = len(wrong_probas)
            allowed_wrong = int(total * wrong_pct)
            if allowed_wrong >= num_wrong:
                return 0.0
            too_many_wrong = num_wrong - allowed_wrong
            idx = min(too_many_wrong, len(wrong_probas) - 1)
            return wrong_probas[idx]

        total = len(predictions)
        max_speedup = max(speedups_wrong) if speedups_wrong else 0
        gmean_speedup = gmean(speedups_wrong) if speedups_wrong else 0
        max_speedup_top_k = max(wrong_speedups_top_k) if wrong_speedups_top_k else 0
        gmean_speedup_top_k = gmean(wrong_speedups_top_k) if wrong_speedups_top_k else 0

        max_speedup_over_default = (
            max(speedups_over_default) if speedups_over_default else 0
        )
        gmean_speedup_over_default = (
            gmean(speedups_over_default) if speedups_over_default else 0
        )
        max_slowdown_over_defalt = (
            min(speedups_over_default) if speedups_over_default else 0
        )
        return {
            "correct": num_correct,
            "wrong": num_wrong,
            "unsure": num_unsure,
            "total": total,
            "wrong_max_speedup": max_speedup,
            "wrong_gmean_speedup": gmean_speedup,
            "top_k_correct": num_correct_top_k,
            "top_k_wrong": num_wrong_top_k,
            "wrong_max_speedup_k": max_speedup_top_k,
            "wrong_gmean_speedup_k": gmean_speedup_top_k,
            "top_k_unsure": top_k_unsure,
            "max_speedup_default": max_speedup_over_default,
            "gmean_speedup_default": gmean_speedup_over_default,
            "max_slowdown_default": max_slowdown_over_defalt,
            "non_default_predictions": num_non_default_predictions,
            "default_better": num_default_better,
        }

    def gen_classes(self, classes, num_spaces):
        """
        If classes=['choice1', 'choice2', 'choice3'], then this function returns
        the following string:
        self.choices.append('choice1')
        self.choices.append('choice2')
        self.choices.append('choice3')
        Used in the generated heuristic to map the index of a choice to its name.
        """
        indent = " " * num_spaces
        return "\n".join([f"{indent}self.choices.append('{c}')" for c in classes])

    def get_default_config(self, row):
        """
        Returns the default config for a given sample. The default config could for example be the config that is
        the chosen by a current handwritten heuristic. This can for example be used in get_unsafe_leaf to
        compare the predicted config with the default config.
        """
        return None

    def gen_predict_fn_def(self):
        """
        Generates the definition of the predict function.
        """
        return "def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:"

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, classes
    ):
        """
        Generates the boilerplate code for the generated heuristic. This includes things like imports, class definition,
        etc.
        """

        boiler_plate = f"""# flake8: noqa: B950
# fmt: off
# This file was generated by AutoHeuristic. Do not modify it manually!
# To regenerate this file, take a look at the steps in the README.md file inside torchgen/_autoheuristic/{opt_name}/
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)

class {heuristic_name}(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

{self.gen_precondition(opt_name, shared_memory, device_capa)}

    def get_confidence_threshold(self) -> float:
        return {threshold}

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
{self.gen_classes(classes, num_spaces=8)}

    def get_name(self) -> str:
        return '{opt_name}'"""
        return boiler_plate

    def add_real_datasets(
        self, datasets, other_datasets, cat_feature2cats, ranking=False
    ):
        """
        Adds datasets specified by the user to the datasets dictionary.
        """
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path,
                    cat_feature2cats=cat_feature2cats,
                    apply_filters=False,
                    add_near_best=ranking,
                )
                datasets[name] = df_other

    def codegen(
        self,
        tree,
        metadata,
        heuristic_name,
        threshold,
        dummy_col_2_col_val,
        unsafe_leaves,
    ):
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
                tree.classes_,
            )
        )
        fn_def = f"\n    {self.gen_predict_fn_def()}"
        lines.append(fn_def)
        tree.codegen(dummy_col_2_col_val, lines, unsafe_leaves)
        self.write_heuristic_to_file(lines, heuristic_name)


if __name__ == "__main__":
    train = AHTrainDecisionTree()
    train.generate_heuristic()
