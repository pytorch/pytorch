## API design guidelines

All functions should accept arguments in the following order. Dots represent any module-specific parameters or buffers, disregarding whether they are used for writing or reading. They should follow the order
```
[weight], [bias], [any buffers], [additional arguments], [optional arugments]
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


### Conversion Steps

1. copy old .c file to lib/THNN/generic 
  - replace static int nn_ -> void THNN_
  - replace lua_State \*L with 'actual' parameters (+ add THNNState\* state)
  - remove any numeric values from return statements, remove the return at the end of the function body
  - remove old luaL_Reg & _init function
2. add forward declarations to generic/THNN.h
3. include the generic/xyz.c file in init.c
4. add functions to ffi.lua
5. copy & adapt lua file: specify module THNN for torch.class(), use THNN.errcheck
6. include module lua file in init.lua
7. add & run unit test to lua/tests/test.lua
