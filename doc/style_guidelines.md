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
void THNN_(HardShrink_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda)
```

### Criterions
```
updateOutput: state, input, target, output, ...
updateGradInput: state, input, target, gradInput, ...
```

e.g.

```C
void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState* state,
          THTensor *input,
          THLongTensor *target,
          THTensor *output,
          THTensor *weights,
          THTensor *total_weight,
          bool sizeAverage)
```

## Code style guide

```C
void THNN_Linear_updateOutput(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
//<- 10 ->
```

All arguments should start on a new line after function name, and they should be indented using 10 spaces.

Use 2 spaces for block indentation.
