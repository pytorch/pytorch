import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class TrainDecisionTreeDepthwiseConv:
    def __init__(self):
        self.features = ["sm", "bs", "ch", "w", "filter", "stride"]
        self.target = "cudnn_speedup_all"
        self.classes = ["false", "true"]
        self.output_file = (
            "../../../aten/src/ATen/autoheuristic/DepthwiseConvHeuristic.h"
        )
        self.opt_name = "depthwise_conv"
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self):
        # Data parameters
        self.parser.add_argument(
            "input_files", type=str, nargs="+", help="Paths to processed CSV files"
        )
        self.parser.add_argument(
            "--tolerance",
            type=float,
            default=0.0,
            help="Tolerance threshold (default: 0.0)",
        )

        # Model parameters
        self.parser.add_argument(
            "--max-depth",
            type=int,
            default=None,
            help="Maximum tree depth (default: None = unlimited)",
        )
        self.parser.add_argument(
            "--max-leaf-nodes",
            type=int,
            default=None,
            help="Maximum number of leaf nodes (default: None = unlimited)",
        )
        self.parser.add_argument(
            "--min-samples-split",
            type=int,
            default=2,
            help="Minimum samples to split a node (default: 2)",
        )
        self.parser.add_argument(
            "--min-samples-leaf",
            type=int,
            default=1,
            help="Minimum samples in leaf node (default: 1)",
        )
        self.parser.add_argument(
            "--criterion",
            type=str,
            default="gini",
            choices=["gini", "entropy", "log_loss"],
            help="Split criterion (default: gini)",
        )

        # Other
        self.parser.add_argument(
            "--seed", type=int, default=42, help="Random seed (default: 42)"
        )

    def parse_args(self):
        return self.parser.parse_args()

    def load_and_prepare_data(self, input_files, tolerance):
        """
        Load data and prepare for binary classification.
        Filters out label 2 (equal performance within tolerance).
        """
        required_columns = [*self.features, self.target]
        dfs = []

        for input_file in input_files:
            if not input_file.endswith(".csv"):
                raise ValueError(
                    f"Invalid file format: {input_file}. Expected CSV file with .csv extension."
                )

            try:
                df_full = pd.read_csv(input_file)
            except Exception as e:
                raise ValueError(f"Failed to read CSV file {input_file}: {e}") from e

            missing_columns = set(required_columns) - set(df_full.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in {input_file}: {sorted(missing_columns)}. "
                    f"Required columns: {required_columns}"
                )
            df = df_full[required_columns]

            if df.isnull().any().any():
                empty_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(
                    f"File {input_file} contains empty cells in columns: {empty_cols}."
                )

            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        features = df[self.features].values
        speedup_values = df[self.target].values
        sample_weights = abs(speedup_values - 1.0) - tolerance

        # Create original labels (0, 1, 2)
        lower_tolerance = 1.0 - tolerance
        upper_tolerance = 1.0 + tolerance
        labels = np.zeros(len(speedup_values), dtype=np.int64)
        labels[speedup_values < lower_tolerance] = 0
        labels[speedup_values > upper_tolerance] = 1
        labels[
            (speedup_values >= lower_tolerance) & (speedup_values <= upper_tolerance)
        ] = 2

        # Filter out label 2 (results within tolerance)
        mask = labels != 2
        features = features[mask]
        labels = labels[mask]
        sample_weights = sample_weights[mask]

        return features, labels, sample_weights

    def create_decision_tree(self, args, features, labels, sample_weights):
        # Create and train decision tree
        print("Model parameters:")
        print(f"  Criterion: {args.criterion}")
        print(
            f"  Max depth: {args.max_depth if args.max_depth else 'None (unlimited)'}"
        )
        print(
            f"  Max leaf nodes: {args.max_leaf_nodes if args.max_leaf_nodes else 'None (unlimited)'}"
        )
        print(f"  Min samples split: {args.min_samples_split}")
        print(f"  Min samples leaf: {args.min_samples_leaf}")
        print()

        model = DecisionTreeClassifier(
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_leaf_nodes=args.max_leaf_nodes,
            random_state=args.seed,
        )

        model.fit(features, labels, sample_weight=sample_weights)

        print("✓ Training complete!")
        print("\nTree statistics:")
        print(f"  Total nodes: {model.tree_.node_count}")
        print(f"  Leaves: {model.tree_.n_leaves}")
        print(f"  Max depth reached: {model.tree_.max_depth}")

        # Evaluate on training data
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        return model.tree_

    def is_leaf_node(self, tree, node_id):
        """Check if a node is a leaf node."""
        return tree.children_left[node_id] == tree.children_right[node_id] == -1

    def get_leaf_class(self, tree, node_id):
        """Get the class label for a leaf node."""
        class_values = tree.value[node_id][0]
        return self.classes[np.argmax(class_values)]

    def codegen_boilerplate(self):
        header = (
            f"""// This file was generated by AutoHeuristic. Do not modify it manually!
// To regenerate this file, take a look at the README.md file inside torchgen/_autoheuristic/{self.opt_name}/"""
            + """
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>

namespace at::native {

template <typename T>
static bool check_cudnn_depthwise_workload_with_filter(const at::Tensor& input, T stride, const at::Tensor& weight) {
  static int sm = at::detail::getCUDAHooks().getDeviceCapability();
  TORCH_INTERNAL_ASSERT(sm != 0, "CUDA not available");

  // 1D conv
  if(at::symint::size<T>(input, 2) == 1 && stride == 1){
    return true;
  }
  // 2D conv
  // only 1/2 stride
  if (stride != 1 && stride != 2) return false;
  // only square filters
  if (at::symint::size<T>(weight, 2) != at::symint::size<T>(weight, 3)) return false;
  auto filter = at::symint::size<T>(weight, 3);
  // only 1/3/5 filter
  if (filter != 1 && filter != 3 && filter != 5) return false;
  // we don't enforce square input but only check width to reduce heuristic space
  if (at::symint::size<T>(input, 3) < 7) return false; // min width 7
  auto w = at::symint::size<T>(input, 3);
  auto ch = at::symint::size<T>(input, 1);
  auto bs = at::symint::size<T>(input, 0);

  // auto-generated heuristic decision tree"""
        )
        footer = "}\n\n} // namespace at::native\n"
        return header, footer

    def codegen(self, tree, lines: list[str]):
        feature_names = self.features

        def codegen_node(node_id, depth):
            """Recursively traverse tree nodes and generate C++ code."""
            indent = "  " * (depth + 1)

            # Handle leaf nodes
            if self.is_leaf_node(tree, node_id):
                class_label = self.get_leaf_class(tree, node_id)
                lines.append(f"{indent}return {class_label};")
                return

            # Internal node - generate condition
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Check if children are leaf nodes for inline formatting
            left_is_leaf = self.is_leaf_node(tree, left_child)
            right_is_leaf = self.is_leaf_node(tree, right_child)

            cpp_condition = f"{feature_name} <= {int(threshold)}"

            if left_is_leaf and right_is_leaf:
                # Both children are leaves - use inline format
                left_class = self.get_leaf_class(tree, left_child)
                right_class = self.get_leaf_class(tree, right_child)
                # Leaves with the same class can be combined
                if left_class == right_class:
                    lines.append(f"{indent}return {left_class};")
                else:
                    lines.append(f"{indent}if ({cpp_condition}) return {left_class};")
                    lines.append(f"{indent}else return {right_class};")
            elif left_is_leaf:
                # Only left child is leaf - inline if, else block
                left_class = self.get_leaf_class(tree, left_child)
                lines.append(f"{indent}if ({cpp_condition}) return {left_class};")
                lines.append(f"{indent}else {{")
                codegen_node(right_child, depth + 1)
                lines.append(f"{indent}}}")
            elif right_is_leaf:
                # Only right child is leaf - if block, inline else
                right_class = self.get_leaf_class(tree, right_child)
                lines.append(f"{indent}if ({cpp_condition}) {{")
                codegen_node(left_child, depth + 1)
                lines.append(f"{indent}}}")
                lines.append(f"{indent}else return {right_class};")
            else:
                # Both children are internal nodes - full if-else blocks
                lines.append(f"{indent}if ({cpp_condition}) {{")
                codegen_node(left_child, depth + 1)
                lines.append(f"{indent}}}")
                lines.append(f"{indent}else {{")
                codegen_node(right_child, depth + 1)
                lines.append(f"{indent}}}")

        codegen_node(0, 0)

    def write_heuristic_to_file(self, lines: list[str]):
        with open(self.output_file, "w") as f:
            f.write("\n".join(lines))

    def generate_heuristic(self):
        self.args = self.parse_args()

        # Set random seed
        np.random.seed(self.args.seed)

        # Load data
        features, labels, sample_weights = self.load_and_prepare_data(
            input_files=self.args.input_files, tolerance=self.args.tolerance
        )

        # Create decision tree
        tree = self.create_decision_tree(self.args, features, labels, sample_weights)

        # Create C++ heuristic from decision tree
        header, footer = self.codegen_boilerplate()
        header += ", tolerance = " + str(self.args.tolerance)
        lines = [header]
        self.codegen(tree, lines)
        lines.append(footer)
        self.write_heuristic_to_file(lines)


if __name__ == "__main__":
    train = TrainDecisionTreeDepthwiseConv()
    train.generate_heuristic()
