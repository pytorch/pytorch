## API design guidelines

Functions should return `void`.

All functions should accept arguments in the following order. `...` represent any module-specific parameters or buffers, disregarding whether they are used for writing or reading. Arguments in `...` below should be ordered like this:
```
[weight], [bias], [any buffers], [additional arguments], [optional arguments]
```

### Modules
```
updateOutput: state, input, output, ...
updateGradInput: state, input, gradOutput, gradInput, ...
accGradParameters: state, input, gradOutput, [gradWeight], [gradBias], ...
```

e.g.
```C
void THNN_(ClassNLLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index)
```

### Criterions
```
updateOutput: state, input, target, output, ...
updateGradInput: state, input, target, gradInput, ...
```

e.g.

```C
void THNN_(ClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index)
```

## Code style guide

```C
void THNN_(GatedLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int dim)
//<- 10 ->
```

All arguments should start on a new line after function name, and they should be indented using 10 spaces.

Use 2 spaces for block indentation.
