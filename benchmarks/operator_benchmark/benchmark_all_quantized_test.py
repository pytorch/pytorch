import operator_benchmark as op_bench
from pt import ( # noqa
    qactivation_test,
    qarithmetic_test,
    qbatchnorm_test,
    qcat_test,
    qcomparators_test,
    qconv_test,
    qgroupnorm_test,
    qinstancenorm_test,
    qinterpolate_test,
    qlayernorm_test,
    qlinear_test,
    qobserver_test,
    qpool_test,
    qrnn_test,
    qtensor_method_test,
    quantization_test,
    qunary_test,
    qembedding_pack_test,
    qembeddingbag_test,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
