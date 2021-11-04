Sample result of DDP benchmark:

Key takeaway:
1) In the forward path, Python impl is 1% faster than DPP Core. Python SYNC and ASYNC are on-par
2) re2 backward, speed ranking: DDPCore > Python Async > Python SYNC.
3) Python ASYNC - backward is [2%-20%] slower than DDPCore, varies by buffer size slightly
   * It's expected since larger buffer size means less parallelization as number of buckets decreases
   * No difference once buffer size is larger than total elements (24.37M) such as 25M, 26M, 27M
3) Python SYNC - backward is 20% slower than DDP Core, independent of buffer size

The following is generated summary from compare_ddp.py

```

=== Summary for buffer_size: 3M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.141   0         210.042   0         210.64    0        210.728    0        212.782   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.221  -0.913365  208.235  -0.860175  208.376  -1.07512  208.418   -1.09622  208.503  -2.01078
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.272  -0.889167  208.269  -0.843666  208.399  -1.06421  208.441   -1.08532  208.526  -2.00021


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.928   0        471.874    0       472.614   0        472.739    0        472.845   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  483.761   2.50722  483.565    2.4776  485.201   2.66322  486.439    2.89801  487.63    3.12684
DDPOption.PYTHON_DDP_SYNC_REDUCTION   566.434  20.0254   567.739   20.3159  571.238  20.8676   572.456   21.0935   574.975  21.5991

=== Summary for buffer_size: 4M ===
DDP: [forward]                           Mean    delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.668   0        210.536   0         211.12    0        211.243    0        213.027   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.519  -1.0202   208.526  -0.954858  208.652  -1.16933  208.688   -1.2098   208.749  -2.00843
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.17   -1.18543  208.188  -1.1155    208.304  -1.33418  208.331   -1.37868  208.375  -2.18384


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                470.809   0        470.785   0        471.277   0        471.3      0        471.341   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  485.663   3.15505  484.757   2.96779  487.478   3.43773  488.309    3.60907  492.57    4.50399
DDPOption.PYTHON_DDP_SYNC_REDUCTION   567.582  20.5547   569.553  20.9794   575.34   22.0812   575.718   22.1554   579.709  22.9916

=== Summary for buffer_size: 5M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.501   0        210.56    0        211.131   0        211.321    0        211.676   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.274  -1.05815  208.286  -1.08026  208.39   -1.29836  208.419   -1.373    208.502  -1.49955
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.205  -1.09092  208.209  -1.11681  208.373  -1.30658  208.413   -1.37579  208.505  -1.49841


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.921    0       471.908   0        472.321   0        472.389    0        472.473   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  491.629    4.176   491.88    4.23226  492.99    4.37604  493.322    4.43122  493.571   4.46542
DDPOption.PYTHON_DDP_SYNC_REDUCTION   568.702   20.5077  570.053  20.7975   574.92   21.7225   576.329   22.003    578.639  22.4703


=== Summary for buffer_size: 6M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.357   0         210.192   0         211.027   0        211.17     0        212.167   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.218  -1.01707   208.208  -0.943888  208.37   -1.2588   208.434   -1.29541  208.527  -1.7153
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.299  -0.978459  208.287  -0.906401  208.47   -1.21135  208.509   -1.26006  208.542  -1.70831


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.916   0        471.911   0        472.311   0        472.452     0       472.552   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  491.337   4.11541  491.197   4.08693  492.112   4.19242  493.681     4.4933  494.128   4.56582
DDPOption.PYTHON_DDP_SYNC_REDUCTION   570.087  20.8028   570.468  20.8848   574.322  21.5982   576.197    21.9589  577.769  22.2656


=== Summary for buffer_size: 7M ===
DDP: [forward]                           Mean    delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.471   0        210.23    0         211.335   0        211.646    0        212.208   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.3    -1.03175  208.283  -0.926544  208.452  -1.3644   208.516   -1.4787   208.683  -1.66069
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.35   -1.00775  208.322  -0.907727  208.457  -1.36216  208.508   -1.48257  208.64   -1.68102


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.447   0        475.652   0        476.054   0        476.204    0        476.343   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  495.606   4.24003  495.646   4.20351  496.405   4.27489  497.102    4.38859  498.552   4.66256
DDPOption.PYTHON_DDP_SYNC_REDUCTION   570.23   19.9356   570.335  19.9059   574.516  20.683    576.003   20.9573   578.729  21.4942


=== Summary for buffer_size: 8M ===
DDP: [forward]                           Mean     delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.426   0         210.404   0        210.867   0        211.004    0        211.632   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.223  -1.0471    208.211  -1.04268  208.378  -1.18039  208.418   -1.22576  208.51   -1.47519
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.441  -0.943627  208.416  -0.94522  208.594  -1.07788  208.632   -1.12397  208.839  -1.31983


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.366    0       471.394   0        471.727   0        471.945     0       472.15    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  503.342    6.7838  503.624   6.83717  504.701   6.99008  504.88      6.9785  505.829   7.13305
DDPOption.PYTHON_DDP_SYNC_REDUCTION   571.185   21.1767  571.836  21.3075   574.946  21.8813   575.869    22.0202  576.892  22.1839


=== Summary for buffer_size: 9M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.666   0        210.605   0        211.053   0        211.409    0        212.955   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.262  -1.14128  208.247  -1.11954  208.39   -1.26134  208.429   -1.40948  208.532  -2.07704
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.249  -1.14753  208.231  -1.12721  208.421  -1.24665  208.462   -1.39398  208.581  -2.05385


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.742    0       475.783   0        476.311   0        476.586    0        477.329   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  505.635    6.2835  505.944   6.33916  507.325   6.51129  507.566    6.50028  509.167   6.67001
DDPOption.PYTHON_DDP_SYNC_REDUCTION   571.18    20.0608  570.698  19.9493   574.136  20.538    574.622   20.5704   577.593  21.0052


=== Summary for buffer_size: 10M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.235   0         210.238   0         210.766   0        210.975    0        211.777   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.364  -0.889884  208.367  -0.890317  208.483  -1.08294  208.537   -1.15571  208.622  -1.48964
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.303  -0.918867  208.306  -0.91942   208.423  -1.11161  208.522   -1.16244  208.663  -1.47023


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.262   0        471.33    0        471.687   0        471.798    0        471.91    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  506.905   7.56332  506.819   7.52959  507.886   7.67448  508.554    7.79049  509.476   7.96033
DDPOption.PYTHON_DDP_SYNC_REDUCTION   568.922  20.723    569.875  20.9079   572.408  21.3534   573.358   21.5261   576.072  22.0724


=== Summary for buffer_size: 11M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90     delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  ---------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.124   0         210.003   0         210.431   0         210.52    0         212.437   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.387  -0.82678   208.382  -0.772006  208.526  -0.905139  208.591  -0.916353  208.661  -1.77738
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.59   -0.730352  208.559  -0.68767   208.795  -0.777346  208.889  -0.775075  209.069  -1.58534


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                470.129   0        470.107   0        470.559   0        470.657    0        471.173   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  512.012   8.90881  510.767   8.64915  516.538   9.77108  517.193    9.88746  517.7     9.87468
DDPOption.PYTHON_DDP_SYNC_REDUCTION   571.999  21.6685   571.894  21.6519   574.109  22.0057   574.949   22.159    576.708  22.3982


=== Summary for buffer_size: 12M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.276   0         210.11    0         210.724   0        211.507    0        212.171   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.166  -1.00303   208.158  -0.929433  208.319  -1.14142  208.352   -1.49137  208.43   -1.76306
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.195  -0.989416  208.204  -0.907414  208.337  -1.13315  208.366   -1.48507  208.409  -1.77307


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                477.764   0        477.682   0        478.799   0        479.107    0        479.753   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  518.928   8.61592  519.858   8.82943  521.836   8.98841  522.852    9.13037  527.113   9.87188
DDPOption.PYTHON_DDP_SYNC_REDUCTION   570.09   19.3244   571.131  19.5631   574.142  19.9129   574.744   19.9613   576.002  20.0623


=== Summary for buffer_size: 13M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.758   0        210.618   0        211.531   0        211.888    0        212.169   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.119  -1.25216  208.104  -1.19357  208.269  -1.54205  208.325   -1.68138  208.382  -1.78527
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.122  -1.25057  208.106  -1.19253  208.245  -1.55357  208.318   -1.68506  208.378  -1.78703


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.435   0        471.443   0        471.977    0       472.104     0       472.534    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  514.153   9.06126  512.027   8.60847  521.519   10.4968  522.174    10.6057  523.45    10.7752
DDPOption.PYTHON_DDP_SYNC_REDUCTION   567.48   20.3729   567.488  20.3725   570.315   20.8354  571.16     20.9817  572.468   21.1486


=== Summary for buffer_size: 14M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.501   0        210.384   0        210.973   0        211.675    0        212.68    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.199  -1.09343  208.211  -1.03275  208.334  -1.25111  208.375   -1.55885  208.471  -1.97944
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.285  -1.0527   208.27   -1.0047   208.434  -1.20361  208.49    -1.50462  208.591  -1.92271


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.438    0       475.465    0       476.153    0       476.409     0       476.711    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  524.349   10.2876  527.475   10.9388  530.305   11.3729  530.555    11.3654  532.441   11.6906
DDPOption.PYTHON_DDP_SYNC_REDUCTION   573.611   20.649   573.127   20.5405  577.294   21.2415  578.41     21.4104  579.276   21.5153


=== Summary for buffer_size: 15M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.461   0        210.391   0        210.934   0        211.032    0        211.317   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.274  -1.03918  208.266  -1.01032  208.43   -1.18727  208.484   -1.20772  208.59   -1.29049
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.227  -1.06146  208.226  -1.02924  208.357  -1.22186  208.422   -1.23705  208.522  -1.32288


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                473.942    0       473.82     0       474.751    0       474.806     0       475.033    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  531.04    12.0476  532.847   12.4576  534.748   12.6376  535.075    12.6935  536.357   12.9093
DDPOption.PYTHON_DDP_SYNC_REDUCTION   572.339   20.7615  572.451   20.816   574.846   21.0838  575.131    21.1299  576.538   21.3679


=== Summary for buffer_size: 16M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                211.386   0        211.473   0        211.924   0        211.985    0        212.352   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.352  -1.43535  208.349  -1.47735  208.534  -1.5998   208.588   -1.60253  208.614  -1.76059
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.166  -1.52366  208.154  -1.56983  208.277  -1.72072  208.32    -1.7291   208.387  -1.86733


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                471.396    0       471.431    0       471.741    0       471.807     0       472.008    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  532.423   12.946   532.087   12.8663  534.801   13.3674  535.472    13.4938  536.216   13.6032
DDPOption.PYTHON_DDP_SYNC_REDUCTION   570.831   21.0937  573.693   21.6918  576.25    22.1538  577.139    22.3251  578.038   22.4638


=== Summary for buffer_size: 17M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.329   0        210.456   0        210.703   0        210.802    0        210.855   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.179  -1.02248  208.168  -1.08692  208.31   -1.13592  208.374   -1.15169  208.406  -1.16142
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.107  -1.05666  208.081  -1.1281   208.288  -1.14616  208.345   -1.16578  208.397  -1.16564


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                485.222   0        485.197   0        486.18    0        486.698    0        488       0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  527.389   8.69042  526.905   8.59613  534.23    9.88322  535.235    9.97259  536.053   9.84694
DDPOption.PYTHON_DDP_SYNC_REDUCTION   575.94   18.6963   576.127  18.7408   578.749  19.04     579.275   19.0214   580.449  18.9443


=== Summary for buffer_size: 18M ===
DDP: [forward]                           Mean    delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.382   0        210.297   0         210.785   0        211.018    0        211.427   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.444  -0.92151  208.417  -0.893902  208.535  -1.06755  208.554   -1.16772  208.694  -1.29247
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.25   -1.01326  208.241  -0.977957  208.412  -1.12568  208.437   -1.22291  208.468  -1.39955


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                480.523    0       480.491    0       481.178    0       481.298     0       481.661    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  537.206   11.796   539.801   12.3435  542.149   12.6713  542.589    12.7344  543.361   12.8099
DDPOption.PYTHON_DDP_SYNC_REDUCTION   561.54    16.8601  561.221   16.8016  563.942   17.2003  568.921    18.2054  572.956   18.9543


=== Summary for buffer_size: 19M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90     delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  ---------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                209.971   0         209.939   0         210.15    0         210.364   0         210.762   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.381  -0.757295  208.33   -0.766203  208.497  -0.7869    208.516  -0.878442  208.721  -0.96848
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.311  -0.790664  208.309  -0.776575  208.457  -0.805903  208.487  -0.892226  208.557  -1.04625


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                478.77     0       478.715    0       479.343    0       479.566     0       479.833    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  540.249   12.8409  538.619   12.5133  547.754   14.2719  548.267    14.3257  549.454   14.5094
DDPOption.PYTHON_DDP_SYNC_REDUCTION   570.878   19.2385  571.444   19.3702  574.852   19.925   575.923    20.0926  581.559   21.2002


=== Summary for buffer_size: 20M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.733   0        210.718   0        211.035   0        211.225    0        212.012   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.228  -1.18869  208.236  -1.17816  208.353  -1.27058  208.376   -1.34851  208.425  -1.69187
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.293  -1.15816  208.277  -1.1586   208.444  -1.22736  208.521   -1.28011  208.615  -1.60261


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.089    0       475.206    0       475.633    0       475.834     0       476.355    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  550.877   15.9523  553.077   16.3869  555.351   16.7605  555.897    16.8258  556.576   16.8405
DDPOption.PYTHON_DDP_SYNC_REDUCTION   574.111   20.8427  574.816   20.9615  578.774   21.6851  580.185    21.9301  586.553   23.1335


=== Summary for buffer_size: 21M ===
DDP: [forward]                           Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.853   0        210.825   0        211.183   0        211.308    0        211.53    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.372  -1.1765   208.361  -1.16885  208.561  -1.24157  208.596   -1.28325  208.692  -1.34181
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.202  -1.25715  208.198  -1.24621  208.368  -1.333    208.407   -1.37313  208.501  -1.43188


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                473.293    0       473.282    0       473.808    0       473.933     0       474.12     0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  538.12    13.697   536.046   13.2614  551.029   16.298   553.37     16.7611  554.301   16.9114
DDPOption.PYTHON_DDP_SYNC_REDUCTION   575.191   21.5296  574.89    21.4687  578.517   22.0996  578.801    22.127   583.41    23.0512


=== Summary for buffer_size: 22M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.444   0         210.341   0         210.864   0        211.106    0        212.947   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.367  -0.986748  208.36   -0.941914  208.51   -1.11658  208.55    -1.21056  208.603  -2.03981
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.511  -0.918347  208.496  -0.877387  208.637  -1.05606  208.656   -1.16059  208.805  -1.94498


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                482.097    0       482.406    0       483.012    0       483.234     0       484.214    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  560.875   16.3408  560.893   16.2698  562.507   16.4582  563.033    16.5135  563.883   16.4531
DDPOption.PYTHON_DDP_SYNC_REDUCTION   575.81    19.4387  575.681   19.3353  578.208   19.7089  580.115    20.0483  583.628   20.531


=== Summary for buffer_size: 23M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90     delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  ---------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.215   0         210.037   0         210.684   0         211.337    0        213.123   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.129  -0.992138  208.114  -0.915331  208.252  -1.15452   208.304   -1.43533  208.426  -2.20396
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.427  -0.850357  208.408  -0.77528   208.616  -0.981803  208.655   -1.26914  208.706  -2.07245


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                481.245    0       481.24     0       481.688    0       482.084     0       482.664    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  564.44    17.2875  566.225   17.6597  568.807   18.0862  569.019    18.033   569.317   17.953
DDPOption.PYTHON_DDP_SYNC_REDUCTION   571.43    18.7401  573.479   19.1671  576.317   19.6453  578.183    19.9341  588.235   21.8725


=== Summary for buffer_size: 24M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.227   0         210.143   0         210.604   0        210.722    0        210.87    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.228  -0.95065   208.224  -0.913135  208.358  -1.06628  208.426   -1.0898   208.654  -1.05091
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.23   -0.949705  208.218  -0.915672  208.435  -1.02967  208.484   -1.06235  208.557  -1.09691


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                478.429    0       478.608    0       479.358    0       479.68      0       480.215    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  553.532   15.6977  548.179   14.536   566.12    18.0998  567.022    18.2083  572.705   19.2602
DDPOption.PYTHON_DDP_SYNC_REDUCTION   572.728   19.71    572.48    19.6134  574.355   19.8176  574.741    19.8175  582.827   21.3679


=== Summary for buffer_size: 25M ===
DDP: [forward]                           Mean    delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.478   0        210.405   0         210.763   0        210.963    0        212.506   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.363  -1.00473  208.344  -0.979519  208.545  -1.05199  208.591   -1.12464  208.724  -1.77946
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.297  -1.03607  208.289  -1.00562   208.429  -1.10732  208.449   -1.19205  208.495  -1.88737


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                478.977    0       478.951    0       479.407    0       479.564     0       480.253    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  569.299   18.8573  573.043   19.6453  574.805   19.8991  575.978    20.1046  583.574   21.5137
DDPOption.PYTHON_DDP_SYNC_REDUCTION   572.347   19.4937  573.636   19.7691  575.768   20.1     576.877    20.292   577.819   20.3156


=== Summary for buffer_size: 26M ===
DDP: [forward]                           Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.289   0         210.129   0         210.915   0        211.356    0        212.755   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.237  -0.975918  208.204  -0.91614   208.435  -1.17554  208.573   -1.31667  208.692  -1.90987
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.108  -1.03731   208.098  -0.966869  208.264  -1.25662  208.335   -1.4294   208.446  -2.02556


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.649    0       475.81     0       476.438     0      476.584     0       476.782    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  574.232   20.726   574.333   20.7063  577.995    21.316  578.914    21.4716  579.428   21.5289
DDPOption.PYTHON_DDP_SYNC_REDUCTION   574.607   20.8051  574.994   20.8453  577.543    21.221  578.45     21.3741  578.905   21.4192


=== Summary for buffer_size: 27M ===
DDP: [forward]                          Mean     delta%     mean     delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  ------  ---------  -------  ---------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                210.39   0         210.259   0         210.829   0        211.385    0        211.908   0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  208.36  -0.964811  208.358  -0.903892  208.496  -1.10633  208.555   -1.33865  208.614  -1.55409
DDPOption.PYTHON_DDP_SYNC_REDUCTION   208.45  -0.922151  208.443  -0.863724  208.581  -1.06633  208.681   -1.27889  208.742  -1.49412


DDP: [backward]                          Mean    delta%     mean    delta%      p90    delta%      p95    delta%%      p99    delta%
------------------------------------  -------  --------  -------  --------  -------  --------  -------  ---------  -------  --------
DDPOption.DDP_CPP_CORE                475.294    0       475.299    0       476.014    0       476.119     0       476.351    0
DDPOption.PYTHON_DDP_ASYNC_REDUCTION  570.313   19.9917  572.485   20.4473  574.882   20.77    575.329    20.8371  576.284   20.9788
DDPOption.PYTHON_DDP_SYNC_REDUCTION   571.216   20.1816  572.884   20.5313  574.973   20.7891  575.856    20.9478  576.293   20.9807

```
