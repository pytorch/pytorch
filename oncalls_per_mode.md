<!-- Here we cut out data points with times less than 1 hour -->
|     | mode        | oncall                 |     time (hours) |
|----:|:------------|:-----------------------|---------:|
|  42 | crossref    | module: unknown        |  2.58015 |
|  80 | default     | module: mkldnn         |  1.04507 |
|  63 | default     | module: cpp-extensions |  1.06332 |
|  70 | default     | module: dynamo         |  1.36428 |
|  55 | default     | module: ProxyTensor    |  1.49594 |
|  73 | default     | module: fx             |  1.73626 |
|  82 | default     | module: mta            |  1.91886 |
|  66 | default     | module: dataloader     |  1.99119 |
|  77 | default     | module: linear algebra |  2.01572 |
|  88 | default     | module: nvfuser        |  2.90216 |
|  94 | default     | module: sparse         |  4.08902 |
|  96 | default     | module: tests          |  4.79804 |
|  54 | default     | NNC                    |  5.57266 |
| 107 | default     | oncall: quantization   |  6.28772 |
| 102 | default     | oncall: jit            |  6.33455 |
|  86 | default     | module: nn             |  7.02614 |
|  75 | default     | module: inductor       | 16.2848  |
|  67 | default     | module: decompositions | 33.3157  |
|  90 | default     | module: primTorch      | 41.5151  |
|  99 | default     | module: unknown        | 65.3485  |
| 112 | distributed | oncall: distributed    | 14.4679  |
| 149 | dynamo      | module: unknown        |  2.35712 |
| 204 | functorch   | module: functorch      | 22.9706  |
| 206 | inductor    | module: inductor       |  1.0735  |
| 209 | inductor    | module: unknown        |  2.5047  |
| 216 | multigpu    | oncall: distributed    |  4.75693 |
| 326 | slow        | module: decompositions |  1.16277 |
| 348 | slow        | module: primTorch      |  1.20082 |
| 360 | slow        | oncall: jit            |  1.41464 |
| 313 | slow        | NNC                    |  1.43947 |
| 346 | slow        | module: nvfuser        |  2.21356 |
| 357 | slow        | module: unknown        |  3.49007 |
