# mypy: ignore-errors

from train import AHTrain

from torch._inductor.fx_passes.pad_mm import pad_mm_operations


class AHTrainPadMM(AHTrain):
    A100_tensor_min_size = 512
    H100_tensor_min_size = 768

    def __init__(self):
        super().__init__()

    def add_base_arguments(self):
        super().add_base_arguments()
        self.parser.add_argument(
            "--gpu",
            type=str,
            default="A100",
            help="Either A100 or H100 to generate GPU specific filter and precondition.",
        )

    def add_new_features(self, results):
        ops = pad_mm_operations()
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
        added_categorical_features = [op.name for op in ops if op.is_categorical]
        return (results, added_categorical_features)

    def filter_df(self, df):
        if self.args.gpu == "A100":
            min_size = AHTrainPadMM.A100_tensor_min_size
        elif self.args.gpu == "H100":
            min_size = AHTrainPadMM.H100_tensor_min_size
        else:
            return df
        # only keep rows where each dim is >= min_size
        df = df[((df["k"] >= min_size) & (df["n"] >= min_size) & (df["m"] >= min_size))]
        return df

    def shape_size_precondition(self, min_size):
        return f"""if context_dict["m"] < {min_size} or context_dict["k"] < {min_size} or context_dict["n"] < {min_size}:
            return False"""

    def add_precondition(self):
        # TODO(AlnisM): Find a better way to support such preconditions
        if self.args.gpu == "A100":
            min_size = AHTrainPadMM.A100_tensor_min_size
        elif self.args.gpu == "H100":
            min_size = AHTrainPadMM.H100_tensor_min_size
        else:
            return ""
        return self.shape_size_precondition(min_size)


if __name__ == "__main__":
    train = AHTrainPadMM()
    train.generate_heuristic()
