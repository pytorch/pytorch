# mypy: ignore-errors

import argparse
import json
import warnings

import pandas as pd  # type: ignore[import-untyped]

from torch._inductor.autoheuristic.autoheuristic_utils import (
    CHOICE_COL,
    get_metadata_str_from_log,
)


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
    Base class for AutoHeuristic training.
    """

    def __init__(self) -> None:
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
        self.parser.add_argument(
            "--save-dot",
            action="store_true",
            help="Export heuristic to graphviz dot.",
        )
        self.parser.add_argument(
            "--ranking",
            type=int,
            default=None,
            help="""
                Makes AutoHeuristic learn a heuristic that ranks choices instead of predicting a single choice.
                The argument is the number of choices the heuristic will provide.
            """,
        )

    def parse_args(self):
        return self.parser.parse_args()

    def parse_log(self, log_path, nrows=None):
        (df, metadata) = self.deserialize_data(log_path)
        numerical_features = metadata["numerical_features"]
        categorical_features = metadata["categorical_features"]
        choices = df[CHOICE_COL].unique().tolist()
        features = numerical_features + categorical_features
        if nrows is not None:
            df = df.head(nrows)
        df = self.filter_df(df)
        return (df, metadata, features, categorical_features, choices)

    def generate_heuristic(self):
        self.args = self.parse_args()
        self.main(
            self.args.dataset,
            self.args.data,
            self.args.nrows,
            self.args.heuristic_name,
            self.args.save_dot,
            self.args.ranking is not None,
        )

    def filter_df(self, df):
        return df

    def add_new_features(self, results):
        return (results, [])

    def add_real_datasets(self, datasets, other_datasets, cat_feature2cats):
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path, cat_feature2cats=cat_feature2cats, apply_filters=False
                )
                datasets[name] = df_other

    def handle_categorical_features(
        self, cat_feature2cats, categorical_features, results
    ):
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
        return (results, cat_feature2cats, dummy_col_2_col_val)

    def gen_precondition(self, opt_name, shared_memory, device_capa):
        return f"""    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == {shared_memory}
            and str(metadata.device_capa) == "{device_capa}"
        )"""

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, dt
    ):
        pass

    def gen_predict_fn_def(self):
        pass

    def write_heuristic_to_file(self, lines, heuristic_name):
        output_file = (
            f"../../../torch/_inductor/autoheuristic/artifacts/_{heuristic_name}.py"
        )
        path = f"{output_file}"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def deserialize_data(self, log_path):
        json_string = get_metadata_str_from_log(log_path)
        metadata = self.deserialize_metadata(json_string)

        df = pd.read_csv(log_path, skiprows=1, on_bad_lines="skip")
        return (df, metadata)

    def deserialize_metadata(self, json_string):
        return json.loads(json_string)


if __name__ == "__main__":
    train = AHTrain()
    train.generate_heuristic()
