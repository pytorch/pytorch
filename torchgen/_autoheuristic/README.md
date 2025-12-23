# AutoHeuristic
AutoHeuristic is a framework that allows one to use results from autotuning to learn a heuristic as a decision tree, that can be generated to code and shipped with compiler.

## How to use AutoHeuristic
In general, the following steps have to performed:
- The AutoHeuristic constructor has to be called.
- A script that runs benchmarks in order to collect training data has to be implemented.
- The train_decision.py (if you want to learn a decision tree) or train_regression.py (if you want to learn a regression tree) script has to be run in order to learn the heuristic and generate it to code.

## Step 1: Calling the AutoHeuristic constructor
Currently, two use cases are supported:

### Use case 1: Local autotuning
When your feedback function is able to immediately return a result, you can just call the AutoHeuristic constructor. This is done e.g. for pad_mm
```
autoheuristic = AutoHeuristic(
    fallback=fallback,
    choices=choices,
    feedback=feedback,
    context=context,
    name=name,
    augment_context=pad_mm_operations(),
    precondition=pad_mm_precondition,
)
```
Here, `feedback` is a function that benchmarks a given choice and returns the execution time. For an example, see: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pad_mm.py.

### Use case 2: Kernel choice selection
If you want to use AutoHeuristic for kernel choice selection, you have to call the AutoHeuristicSelectAlgorithm constructor. This is done e.g. for mixed_mm
```
autoheuristic = AutoHeuristicSelectAlgorithm(
    fallback=fallback,
    choices=choices,
    input_nodes=input_nodes,
    context=context,
    name=name,
    augment_context=ops,
    precondition=precondition,
)
```
This call has to be followed by a call to `autotune_select_algorithm()`,
```
autotune_select_algorithm(name, choices, input_nodes, layout)
```
Note that `choices`, `input_nodes`, and `name` in the `AutoHeuristicSelectAlgorithm()` and `autotune_select_algorithm()` calls have to match when you want to use AutoHeuristic to collect data.

For an example, see: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py

## Step 2: Collecting training data
After adding the call to the AutoHeuristic constructor, you need to collect training data in order to learn a heuristic. Let's say you have a script `run.py` that triggers the AutoHeuristic constructor that you just added. Run the following command in order to store data into file `train.txt`:
```
TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH="train.txt" \
  TORCHINDUCTOR_AUTOHEURISTIC_COLLECT="pad_mm" python run.py
```
Replace "pad_mm" with the name you provided in the call to the AutoHeuristic constructor.

AutoHeuristic provides a `BenchmarkRunner` class (https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/benchmark_runner.py) that simplifies the process of collecting data. To use it, create a new class that subclasses `BenchmarkRunner`, and implements the `run_benchmark()` and `create_input()` methods.

These examples might be helpful:
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/pad_mm/gen_data_pad_mm.py
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/mixed_mm/gen_data_mixed_mm.py


## Step 3: Learning a heuristic and using it
Once you have collected enough training data, you are ready to learn a heuristic:
```
python torchgen/_autoheuristic/train_decision.py train.txt --heuristic-name SimpleHeuristic
```
will learn a heuristic and generate it to `torch/_inductor/autoheuristic/artifacts/_SimpleHeuristic.py`.

You can now use your learned heuristic:
```
TORCHINDUCTOR_AUTOHEURISTIC_USE="pad_mm" python run.py
```
Here, you again have to replace "pad_mm" with the name you provided in the call to the AutoHeuristic constructor.

Instead of just running the `train_decision.py` script, you probably want to customize the training process in some way. To do this, create a new class that subclasses `AHTrainDecision` and override methods you want to customize. Here are some examples:
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/mixed_mm/train_decision_mixedmm.py
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/pad_mm/train_decision_pad_mm.py

## Other

### How do I specify features that the heuristic is going to use to make a decision?
The AutoHeuristic constructor requires a `context` argument of type `AHContext`, which will contain all features. You specify features in the following way:
```
context = AHContext()

# adding numerical features
context.add_feature("m", mat1.shape[0])
context.add_feature("k", mat1.shape[1])

# adding a categorical feature
context.add_feature("mat1_dtype", mat1.dtype, is_categorical=True)
```

You might want to use features that are a combination of other features, such as `m*k`. You can of course add such features in the same way as above, i.e.,
```
context.add_feature("m*k", mat1.shape[0] * mat1.shape[1])
```
but AutoHeuristic also provides a way to 'augment' features. Augmented features are not stored when data is collected, instead they are created before a heuristic is learned, or before a learned heuristic is used. You can specify such augmented features by creating a list of `AHOperation` objects:
```
def m_times_k(data: Any) -> float:
    return data['m'] * data['k']

m_times_k_op = AHOperation("m*k', m_times_k)
ah_operations = [m_times_k_op]

# specify augmented features by setting `augment_context` to `ah_operations`
autoheuristic = AutoHeuristic(..., augment_context=ah_operations, ...)
```

Note that you also have to specify these operations when you want to learn a heuristic. Look at the `add_new_features()` method in these examples, to see how it is done:
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/mixed_mm/train_decision_mixedmm.py
- https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/pad_mm/train_decision_pad_mm.py

### Where has AutoHeuristic already been used?
Take a look at the following PRs in which AutoHeuristic has enabled for various optimizations.
Looking at these examples may be helpful if you want to use AutoHeuristic yourself.
- pad_mm: https://github.com/pytorch/pytorch/pull/128643
- mixed_mm:
    - Enabling of AutoHeuristic: https://github.com/pytorch/pytorch/pull/131610
    - Script to collect data: https://github.com/pytorch/pytorch/pull/131611
    - A100 heuristic: https://github.com/pytorch/pytorch/pull/131613
    - H100 heuristic: https://github.com/pytorch/pytorch/pull/132685
- flex_attention: https://github.com/pytorch/pytorch/pull/130398
- mm (heuristic for ranking choices):
    - https://github.com/pytorch/pytorch/pull/131615
    - https://github.com/pytorch/pytorch/pull/131617
    - https://github.com/pytorch/pytorch/pull/131705
    - https://github.com/pytorch/pytorch/pull/131714
