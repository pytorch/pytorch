|    | oncall                       | time (seconds) |   time in hours |
|---:|:-----------------------------|------------:|----------------:|
|  7 | module: complex              |      4.2445 |      0.00117903 |
| 34 | module: numpy                |      8.0795 |      0.00224431 |
| 11 | module: cuda graphs          |     22.9653 |      0.00637926 |
| 56 | oncall: run_tests            |     32.9938 |      0.00916495 |
| 38 | module: pytree               |     36.5362 |      0.0101489  |
| 21 | module: fx.passes            |     44.6437 |      0.012401   |
| 57 | oncall: unsure               |     47.2767 |      0.0131324  |
| 31 | module: named tensor         |     50.8335 |      0.0141204  |
| 18 | module: functionalization    |     54.446  |      0.0151239  |
|  3 | module: __torch_function__   |     88.2783 |      0.0245218  |
|  2 | module: __torch_dispatch__   |     94.0647 |      0.0261291  |
| 32 | module: nestedtensor         |     98.158  |      0.0272661  |
| 47 | module: vmap                 |     99.3437 |      0.0275955  |
| 23 | module: intel                |    117.46   |      0.0326277  |
| 51 | oncall: package/deploy       |    118.766  |      0.0329906  |
|  5 | module: ci                   |    169.533  |      0.0470924  |
| 39 | module: scatter & gather ops |    173.49   |      0.0481915  |
|  6 | module: codegen              |    212.934  |      0.0591482  |
| 17 | module: fft                  |    309.687  |      0.0860243  |
| 55 | oncall: r2p                  |    319.817  |      0.0888379  |
| 53 | oncall: pt2                  |    358.175  |      0.0994931  |
| 26 | module: meta tensors         |    364.213  |      0.10117    |
| 44 | module: type promotion       |    438.043  |      0.121679   |
| 45 | module: typing               |    556.698  |      0.154638   |
| 28 | module: mps                  |    619.012  |      0.171948   |
|  8 | module: cpp                  |    650.101  |      0.180584   |
| 52 | oncall: profiler             |    684.902  |      0.190251   |
| 25 | module: masked operators     |    954.113  |      0.265031   |
| 14 | module: dispatch             |   1398.59   |      0.388498   |
| 15 | module: distributions        |   1714.31   |      0.476197   |
| 42 | module: tensor creation      |   2076.32   |      0.576756   |
|  4 | module: autograd             |   2674.89   |      0.743024   |
| 40 | module: serialization        |   2689.94   |      0.747204   |
| 50 | oncall: mobile               |   2739.42   |      0.76095    |
| 10 | module: cuda                 |   3297.72   |      0.916032   |
| 36 | module: optimizer            |   4491.57   |      1.24766    |
|  9 | module: cpp-extensions       |   4903.15   |      1.36199    |
| 30 | module: multiprocessing      |   5078.05   |      1.41057    |
| 16 | module: dynamo               |   6249.94   |      1.73609    |
| 27 | module: mkldnn               |   6351.32   |      1.76426    |
|  1 | module: ProxyTensor          |   6536.86   |      1.81579    |
| 29 | module: mta                  |   7497.87   |      2.08274    |
| 24 | module: linear algebra       |   7891.06   |      2.19196    |
| 20 | module: fx                   |   9397.1    |      2.61031    |
| 12 | module: dataloader           |  10100.9    |      2.80581    |
| 41 | module: sparse               |  17087.1    |      4.74642    |
| 35 | module: nvfuser              |  18420.1    |      5.11669    |
| 43 | module: tests                |  20157.7    |      5.59935    |
|  0 | NNC                          |  27275.6    |      7.57656    |
| 33 | module: nn                   |  28236.1    |      7.84335    |
| 54 | oncall: quantization         |  30410.6    |      8.4474     |
| 49 | oncall: jit                  |  31634.1    |      8.78724    |
| 48 | oncall: distributed          |  69270.7    |     19.2419     |
| 22 | module: inductor             |  69664      |     19.3511     |
| 19 | module: functorch            |  82694.1    |     22.9706     |
| 13 | module: decompositions       | 124788      |     34.6633     |
| 37 | module: primTorch            | 155840      |     43.289      |
| 46 | module: unknown              | 275206      |     76.4462     |
