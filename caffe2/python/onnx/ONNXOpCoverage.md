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
|Acos|Yes|OK|&#x1F49A;OK|
|Add|Yes|OK|&#x1F49A;OK|
|And|Yes|Support int tensor, but no bool tensor|&#x1F49A;OK|
|ArgMax|||&#x1F49A;OK|
|ArgMin|||&#x1F49A;OK|
|Asin|||&#x1F49A;OK|
|Atan|||&#x1F49A;OK|
|AveragePool||OK|&#x1F49A;OK|
|BatchNormalization||OK|&#x1F49A;OK|
|Cast|Yes||&#x1F494;Need extension|
|Ceil|Yes||&#x1F49A;OK|
|Clip|Yes|OK|&#x1F49A;OK|
|Concat|Yes|OK|&#x1F49A;OK|
|Constant|Yes|OK|&#x1F49B;Special handling|
|Conv|Yes|OK|&#x1F49A;OK|
|ConvTranspose|Yes||&#x1F49A;OK, under enhancement|
|Cos|Yes|OK|&#x1F49A;OK|
|DepthToSpace|Yes||&#x1F494;No op|
|Div|Yes|OK|&#x1F49A;OK|
|Dropout|Yes|OK|&#x1F49A;OK|
|Elu|Yes|OK|&#x1F49A;OK|
|Equal|Yes|OK|&#x1F49A;OK|
|Exp|Yes|OK|&#x1F49A;OK|
|Flatten|Yes|OK|&#x1F49A;OK|
|Floor|Yes||&#x1F49A;OK|
|GRU|||&#x1F49A;|
|Gather|Yes|OK|&#x1F49B;C2 only support axis=0 or 1, under development|
|Gemm|Yes|OK|&#x1F49B;C2 use FC or MatMul + Add|
|GlobalAveragePool|Yes|No direct mapping|&#x1F49A;OK|
|GlobalLpPool|||&#x1F494;No mapping yet|
|GlobalMaxPool|||&#x1F49A;OK|
|Greater|Yes||&#x1F49A;OK|
|HardSigmoid|Yes||&#x1F494;No op|
|Hardmax|Yes||&#x1F494;No op|
|InstanceNormalization|||&#x1F49A;OK|
|LRN||OK|&#x1F49A;OK|
|LSTM|||&#x1F49A;OK|
|LeakyRelu|Yes|OK|&#x1F49A;OK|
|Less|Yes||&#x1F49A;OK|
|Log|Yes|OK|&#x1F49A;OK|
|LogSoftmax||OK|&#x1F49A;No op, translated in onnx-caffe2|
|LpNormalization|||&#x1F494;ONNX and C2 have different definition|
|LpPool|||&#x1F49A;Should be LpPool, no tests|
|MatMul|Yes|OK|&#x1F49A;OK|
|Max|Yes|OK|&#x1F49A;OK|
|MaxPool||OK|&#x1F49A;OK|
|MaxRoiPool|||&#x1F494;No mapping yet|
|Mean|||&#x1F49A;OK, need broadcasting support|
|Min|Yes|OK|&#x1F49A;OK, need broadcasting support|
|Mul|Yes|OK|&#x1F49A;OK, need broadcasting support|
|Multinomial|Yes|OK|&#x1F494;no op|
|Neg|Yes|OK|&#x1F49A;OK|
|Not|Yes||&#x1F49A;OK|
|Or|Yes||&#x1F49A;OK|
|PRelu|Yes|OK|&#x1F49B;Need to enhance C2 implementation|
|Pad|Yes|OK|&#x1F49A;OK|
|Pow|Yes|OK|&#x1F49A;OK|
|RNN|||&#x1F49A;OK|
|RandomNormal|||&#x1F494;No op|
|RandomNormalLike|||&#x1F494;No op|
|RandomUniform|||&#x1F494;No op|
|RandomUniformLike|||&#x1F494;No op|
|Reciprocal|Yes||&#x1F49A;Use Pow to implement|
|ReduceL1|||&#x1F494;No op|
|ReduceL2|||&#x1F494;No op|
|ReduceLogSum|||&#x1F494;No op|
|ReduceLogSumExp|||&#x1F494;No op|
|ReduceMax|||&#x1F49A;OK|
|ReduceMean|||&#x1F49A;OK|
|ReduceMin|||&#x1F49A;OK|
|ReduceProd|||&#x1F49A;OK|
|ReduceSum|||&#x1F49A;OK|
|ReduceSumSquare|||&#x1F494;No op|
|Relu|Yes|OK|&#x1F49A;OK|
|Reshape|Yes|OK|&#x1F49A;OK|
|Selu|Yes|OK|&#x1F49A;OK|
|Sigmoid|Yes|OK|&#x1F49A;OK|
|Sin|Yes|OK|&#x1F49A;OK|
|Size|Yes|OK|&#x1F49A;OK|
|Slice|Yes|OK|&#x1F494;ScatterAssign + Cast, very hacky implementation, Slice in C2 only supports one dimension|
|Softmax|Yes|OK|&#x1F494;Axis and dim has different semantics|
|Softplus|Yes|OK|&#x1F49A;OK|
|Softsign|Yes||&#x1F49A;OK|
|SpaceToDepth|||&#x1F494;No op|
|Split|Yes|OK|&#x1F49A;OK|
|Sqrt|Yes||&#x1F49A;OK|
|Squeeze|Yes||&#x1F49A;OK|
|Sub|Yes|OK|&#x1F49A;OK|
|Sum|Yes|OK|&#x1F49A;OK, need broadcasting support|
|Tanh|Yes|OK|&#x1F49A;OK|
|Tile||OK|&#x1F49B;OK, need some enhance|
|TopK||OK|&#x1F49A;OK|
|Transpose|Yes|OK|&#x1F49A;OK|
|Upsample|||&#x1F49B;No bilinear|
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
