# Tracking why operators are not covered
[ONNX backend test script](https://github.com/onnx/onnx-caffe2/blob/master/tests/onnx_backend_test.py)
reports the coverage on the operators and attributes. But we have various of reasons for the missing test coverage on operators.
This doc keeps tracking why operators are not covered by the testcases.

- &#x1F49A; The ONNX operator can map to a Caffe2 operator.
- &#x1F49B; The solution is not perfect/finished, for example, the operator can map to a combination of Caffe2 operators.
- &#x1F494; Hard to find a solution with existing Caffe2 operators.

| Operator | Test Coverage | PyTorch | Caffe2 |
|---|:--:|:---:|:---:|
|Abs|Yes|OK|&#x1F49A;OK|
|Add|Yes|OK|&#x1F49A;OK|
|And|Yes|Support int tensor, but no bool tensor|&#x1F49A;OK|
|ArgMax|||&#x1F49B;About to land|
|ArgMin|||&#x1F49B;About to land|
|AveragePool||OK|&#x1F49A;OK|
|BatchNormalization||OK|&#x1F49A;OK|
|Cast|Yes||&#x1F494;Need extendtion|
|Ceil|Yes||&#x1F49A;OK|
|Clip|Yes|OK|&#x1F49A;OK|
|Concat|Yes|OK|&#x1F49A;OK|
|Constant|Yes|OK|&#x1F49B;Special handling|
|Conv|Yes|OK|&#x1F49A;OK|
|ConvTranspose|Yes||&#x1F49A;OK|
|DepthToSpace|Yes||&#x1F494;No op|
|Div|Yes|OK|&#x1F49A;OK|
|Dropout|Yes|OK|&#x1F49A;OK|
|Elu|Yes|OK|&#x1F49A;OK|
|Equal|Yes|OK|&#x1F49A;OK|
|Exp|Yes|OK|&#x1F49A;OK|
|Flatten|Yes|OK|&#x1F49A;OK|
|Floor|Yes||&#x1F49A;OK|
|GRU|||&#x1F49B;Under development|
|Gather|Yes|OK|&#x1F49B;C2 only support axis=0 or 1, under development|
|Gemm|Yes|OK|&#x1F49B;C2 use FC or MatMul + Add|
|GlobalAveragePool|Yes|No direct mapping|&#x1F49A;OK|
|GlobalLpPool|||&#x1F494;No op|
|GlobalMaxPool|||&#x1F49A;OK|
|Greater|Yes||&#x1F49A;OK|
|HardSigmoid|Yes||&#x1F494;No op|
|Hardmax|Yes||&#x1F494;No op|
|InstanceNormalization|||&#x1F49A;OK|
|LRN||OK|&#x1F49A;OK|
|LSTM|||&#x1F49B;Under development|
|LeakyRelu|Yes|OK|&#x1F49A;OK|
|Less|Yes||&#x1F49A;|
|Log|Yes|OK|&#x1F49A;OK|
|LogSoftmax||OK|&#x1F49B;No op, translated in onnx-caffe2|
|LpNormalization|||&#x1F49A;Should be LpNorm, no tests|
|LpPool|||&#x1F49A;Should be LpPool, no tests|
|MatMul|Yes|OK|&#x1F49A;OK|
|Max|Yes|OK|&#x1F49A;OK|
|MaxPool||OK|&#x1F49A;OK|
|MaxRoiPool|||&#x1F494;No op|
|Mean|||&#x1F49A;OK|
|Min|Yes|OK|&#x1F49A;OK|
|Mul|Yes|OK|&#x1F49A;OK|
|Neg|Yes|OK|&#x1F49A;OK|
|Not|Yes||&#x1F49A;OK|
|Or|Yes||&#x1F49A;OK|
|PRelu|Yes|OK|&#x1F49A;OK|
|Pad|Yes|OK|&#x1F49A;OK|
|Pow|Yes|OK|&#x1F49A;OK|
|RNN|||&#x1F49B;Under development|
|RandomNormal|||&#x1F494;No op|
|RandomNormalLike|||&#x1F494;No op|
|RandomUniform|||&#x1F494;No op|
|RandomUniformLike|||&#x1F494;No op|
|Reciprocal|Yes||&#x1F49B;Use Pow to implement|
|ReduceL1|||&#x1F494;No op|
|ReduceL2|||&#x1F494;No op|
|ReduceLogSum|||&#x1F494;No op|
|ReduceLogSumExp|||&#x1F494;No op|
|ReduceMax|||&#x1F494;No op|
|ReduceMean|||&#x1F494;No op|
|ReduceMin|||&#x1F494;No op|
|ReduceProd|||&#x1F494;No op|
|ReduceSum|||&#x1F494;No op|
|ReduceSumSquare|||&#x1F494;No op|
|Relu|Yes|OK|&#x1F49A;OK|
|Reshape|Yes|OK|&#x1F49A;OK|
|Selu|Yes|OK|&#x1F49A;OK|
|Sigmoid|Yes|OK|&#x1F49A;OK|
|Slice|Yes|OK|&#x1F494;ScatterAssign + Cast, very hacky implementaion, Slice in C2 only supports one dimension|
|Softmax|Yes|OK|&#x1F494;Axis and dim has different semantics|
|Softplus|Yes|OK|&#x1F49A;OK|
|Softsign|Yes||&#x1F49A;OK|
|SpaceToDepth|||&#x1F494;No op|
|Split|Yes|OK|&#x1F49A;OK|
|Sqrt|Yes||&#x1F49A;OK|
|Squeeze|Yes||&#x1F49A;OK|
|Sub|Yes|OK|&#x1F49A;OK|
|Sum|Yes|OK|&#x1F49A;OK|
|Tanh|Yes|OK|&#x1F49A;OK|
|Tile|||&#x1F49B;OK, no tests|
|Transpose|Yes|OK|&#x1F49A;OK|
|Xor|Yes||&#x1F49A;OK|
|experimental ATen|||&#x1F49A;OK|
|experimental Affine|||&#x1F494;No op|
|experimental ConstantFill|||&#x1F49A;OK|
|experimental Crop|||&#x1F494;No op|
|experimental FC|||&#x1F49A;OK|
|experimental GRUUnit|||&#x1F49A;OK, no tests|
|experimental GivenTensorFill|||&#x1F49A;OK|
|experimental Identity|||&#x1F49A;OK|
|experimental ImageScaler|||&#x1F494;No op|
|experimental MeanVarianceNormalization|||&#x1F494;No op|
|experimental ParametricSoftplus|||&#x1F494;No op|
|experimental Scale|||&#x1F49A;OK|
|experimental ScaledTanh|||&#x1F494;No op|
|experimental ThresholdedRelu|Yes||&#x1F49A;OK|
|experimental Upsample|||&#x1F494;No bilinear|
