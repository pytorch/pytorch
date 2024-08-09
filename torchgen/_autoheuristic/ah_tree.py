# mypy: ignore-errors
import numpy as np
from sklearn.tree import _tree


class DecisionTreeNode:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        num_samples=None,
        node_id=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.num_samples = num_samples
        self.id = node_id

    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, sklearn_tree, feature_names):
        self.feature_names = feature_names
        self.root = self._convert_sklearn_tree(sklearn_tree.tree_)
        self.classes_ = sklearn_tree.classes_

    def _convert_sklearn_tree(self, sklearn_tree, node_id=0):
        value = sklearn_tree.value[node_id][0]
        num_samples = sklearn_tree.n_node_samples[node_id]
        if sklearn_tree.feature[node_id] != _tree.TREE_UNDEFINED:
            feature_index = sklearn_tree.feature[node_id]
            feature = self.feature_names[feature_index]
            left = self._convert_sklearn_tree(
                sklearn_tree, sklearn_tree.children_left[node_id]
            )
            right = self._convert_sklearn_tree(
                sklearn_tree, sklearn_tree.children_right[node_id]
            )
            return DecisionTreeNode(
                feature=feature,
                threshold=sklearn_tree.threshold[node_id],
                left=left,
                right=right,
                value=value,
                num_samples=num_samples,
                node_id=node_id,
            )
        else:
            return DecisionTreeNode(
                value=value, num_samples=num_samples, node_id=node_id
            )

    def prune(self, df, target_col, k):
        self.root = self._prune_tree(self.root, df, target_col, k)

    def _prune_tree(self, node, df, target_col, k):
        if node.is_leaf():
            return node

        left_df = df[df[node.feature] <= node.threshold]
        right_df = df[df[node.feature] > node.threshold]

        left_counts = left_df[target_col].nunique()
        right_counts = right_df[target_col].nunique()

        if left_counts < k or right_counts < k:
            return DecisionTreeNode(value=node.value)

        node.left = self._prune_tree(node.left, left_df, target_col, k)
        node.right = self._prune_tree(node.right, right_df, target_col, k)

        return node

    def to_dot(self):
        dot = "digraph DecisionTree {\n"
        dot += '    node [fontname="helvetica"];\n'
        dot += '    edge [fontname="helvetica"];\n'
        dot += self._node_to_dot(self.root)
        dot += "}"
        return dot

    def _node_to_dot(self, node, parent_id=0, edge_label=""):
        if node is None:
            return ""

        node_id = id(node)

        # Format value array with line breaks
        value_str = self._format_value_array(node.value, node.num_samples)

        if node.is_leaf():
            label = value_str
            shape = "box"
        else:
            feature_name = f"{node.feature}"
            label = f"{feature_name} <= {node.threshold:.2f}\\n{value_str}"
            shape = "oval"

        dot = f'    {node_id} [label="{label}", shape={shape}];\n'

        if parent_id != 0:
            dot += f'    {parent_id} -> {node_id} [label="{edge_label}"];\n'

        if not node.is_leaf():
            dot += self._node_to_dot(node.left, node_id, "<=")
            dot += self._node_to_dot(node.right, node_id, ">")

        return dot

    def _format_value(self, num):
        if num == 0:
            return "0"
        return f"{num:.2f}"

    def _format_value_array(self, value, num_samples, max_per_line=5):
        flat_value = value.flatten()
        formatted = [self._format_value(v) for v in flat_value]
        lines = [
            formatted[i : i + max_per_line]
            for i in range(0, len(formatted), max_per_line)
        ]
        return f"num_samples={num_samples}\\n" + "\\n".join(
            [", ".join(line) for line in lines]
        )

    def predict(self, X):
        predictions = [self._predict_single(x) for _, x in X.iterrows()]
        return np.array(predictions)

    def predict_proba(self, X):
        return np.array([self._predict_proba_single(x) for _, x in X.iterrows()])

    def _get_leaf(self, X):
        node = self.root
        while not node.is_leaf():
            if X[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _predict_single(self, x):
        node = self._get_leaf(x)
        # map index to class name
        return self.classes_[np.argmax(node.value)]

    def _predict_proba_single(self, x):
        node = self._get_leaf(x)
        return node.value

    def apply(self, X):
        ids = [self._apply_single(x) for _, x in X.iterrows()]
        return np.array(ids)

    def _apply_single(self, x):
        node = self._get_leaf(x)
        return node.id

    def codegen(self, dummy_col_2_col_val, lines, unsafe_leaves):
        def codegen_node(node, depth):
            indent = "    " * (depth + 1)
            if node.is_leaf():
                lines.append(handle_leaf(node, indent, unsafe_leaves))
            else:
                name = node.feature
                threshold = node.threshold
                if name in dummy_col_2_col_val:
                    (orig_name, value) = dummy_col_2_col_val[name]
                    predicate = f"{indent}if str(context.get_value('{orig_name}')) != '{value}':"
                    assert (
                        threshold == 0.5
                    ), f"expected threshold to be 0.5 but is {threshold}"
                else:
                    predicate = (
                        f"{indent}if context.get_value('{name}') <= {threshold}:"
                    )
                lines.append(predicate)
                codegen_node(node.left, depth + 1)
                lines.append(f"{indent}else:")
                codegen_node(node.right, depth + 1)

        def handle_leaf(node, indent, unsafe_leaves):
            """
            This generates the code for a leaf node in the decision tree. If the leaf is unsafe, the learned heuristic
            will return "unsure" (i.e. None).
            """
            if node.id in unsafe_leaves:
                return f"{indent}return None"
            class_probas = node.value
            return f"{indent}return {best_probas_and_indices(class_probas)}"

        def best_probas_and_indices(class_probas):
            """
            Given a list of tuples (proba, idx), this function returns a string in which the tuples are
            sorted by proba in descending order. E.g.:
            Given class_probas=[(0.3, 0), (0.5, 1), (0.2, 2)]
            this function returns
            "[(0.5, 1), (0.3, 0), (0.2, 2)]"
            """
            # we generate a list of tuples (proba, idx) sorted by proba in descending order
            # idx is the index of a choice
            # we only generate a tuple if proba > 0
            probas_indices_sorted = sorted(
                [
                    (proba, index)
                    for index, proba in enumerate(class_probas)
                    if proba > 0
                ],
                key=lambda x: x[0],
                reverse=True,
            )
            probas_indices_sorted_str = ", ".join(
                f"({value:.3f}, {index})" for value, index in probas_indices_sorted
            )
            return f"[{probas_indices_sorted_str}]"

        return codegen_node(self.root, 1)
