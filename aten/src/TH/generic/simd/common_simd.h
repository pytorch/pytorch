#ifndef COMMON_SIMD_H
#define COMMON_SIMD_H

/* Weights */
#define LOAD_WEIGHT(q, simd_type, inst_var) _m ## simd_type ## inst_var(*(q))

#define DECLARE_WEIGHTS(simd_type) \
__ ## simd_type weight0; \
__ ## simd_type weight1; \
__ ## simd_type weight2; \
__ ## simd_type weight3; \
__ ## simd_type weight4;

#define LOAD_WEIGHTS(k, simd_type, inst_var) \
weight0 = LOAD_WEIGHT(weight + 5 * 0 + k, simd_type, inst_var); \
weight1 = LOAD_WEIGHT(weight + 5 * 1 + k, simd_type, inst_var); \
weight2 = LOAD_WEIGHT(weight + 5 * 2 + k, simd_type, inst_var); \
weight3 = LOAD_WEIGHT(weight + 5 * 3 + k, simd_type, inst_var); \
weight4 = LOAD_WEIGHT(weight + 5 * 4 + k, simd_type, inst_var);

/* Inputs declare */
#define DECLARE_INPUT_0(i) \
float* input0 = image + i; \

#define DECLARE_INPUT_1() \
float* input1 = input0 + inputStride; \
float* input2 = input1 + inputStride; \
float* input3 = input2 + inputStride; \
float* input4 = input3 + inputStride;

#define DECLARE_INPUT_2() \
DECLARE_INPUT_1() \
float* input5 = input4 + inputStride;

#define DECLARE_INPUT_4() \
DECLARE_INPUT_2() \
float* input6 = input5 + inputStride; \
float* input7 = input6 + inputStride;

#define DECLARE_INPUT_5() \
DECLARE_INPUT_4() \
float* input8 = input7 + inputStride;

#define DECLARE_INPUT_6() \
DECLARE_INPUT_5() \
float* input9 = input8 + inputStride;

#define DECLARE_INPUT_7() \
DECLARE_INPUT_6() \
float* inputA = input9 + inputStride;

#define DECLARE_INPUT_8() \
DECLARE_INPUT_7() \
float* inputB = inputA + inputStride;


/* Inputs increment */
#define INC_INPUT_1()\
input0++; \
input1++; \
input2++; \
input3++; \
input4++; \

#define INC_INPUT_2()\
INC_INPUT_1() \
input5++;

#define INC_INPUT_4()\
INC_INPUT_2() \
input6++; \
input7++;

#define INC_INPUT_5()\
INC_INPUT_4() \
input8++;

#define INC_INPUT_6()\
INC_INPUT_5() \
input9++;

#define INC_INPUT_7()\
INC_INPUT_6() \
inputA++;

#define INC_INPUT_8()\
INC_INPUT_7() \
inputB++;

/* Outputs declare */
#define DECLARE_OUTPUT_1() \
float* output0 = output;

#define DECLARE_OUTPUT_2() \
DECLARE_OUTPUT_1() \
float* output1 = output0 + outputStride;

#define DECLARE_OUTPUT_4() \
DECLARE_OUTPUT_2() \
float* output2 = output1 + outputStride; \
float* output3 = output2 + outputStride;

#define DECLARE_OUTPUT_5() \
DECLARE_OUTPUT_4() \
float* output4 = output3 + outputStride;

#define DECLARE_OUTPUT_6() \
DECLARE_OUTPUT_5() \
float* output5 = output4 + outputStride;

#define DECLARE_OUTPUT_7() \
DECLARE_OUTPUT_6() \
float* output6 = output5 + outputStride;

#define DECLARE_OUTPUT_8() \
DECLARE_OUTPUT_7() \
float* output7 = output6 + outputStride;

/* Outputs increment */
#define INC_OUTPUT_1(x) \
output0 += x;

#define INC_OUTPUT_2(x) \
INC_OUTPUT_1(x) \
output1 += x;

#define INC_OUTPUT_4(x) \
INC_OUTPUT_2(x) \
output2 += x; \
output3 += x;

#define INC_OUTPUT_5(x) \
INC_OUTPUT_4(x) \
output4 += x;

#define INC_OUTPUT_6(x) \
INC_OUTPUT_5(x) \
output5 += x;

#define INC_OUTPUT_7(x) \
INC_OUTPUT_6(x) \
output6 += x;

#define INC_OUTPUT_8(x) \
INC_OUTPUT_7(x) \
output7 += x;

/* Image declare */
#define DECLARE_IMAGE_1(simd_type) \
__ ## simd_type image0; \
__ ## simd_type image1; \
__ ## simd_type image2; \
__ ## simd_type image3; \
__ ## simd_type image4;

#define DECLARE_IMAGE_2(simd_type) \
DECLARE_IMAGE_1(simd_type) \
__ ## simd_type image5;

#define DECLARE_IMAGE_4(simd_type) \
DECLARE_IMAGE_2(simd_type) \
__ ## simd_type image6; \
__ ## simd_type image7;

#define DECLARE_IMAGE_5(simd_type) \
DECLARE_IMAGE_4(simd_type) \
__ ## simd_type image8;

#define DECLARE_IMAGE_6(simd_type) \
DECLARE_IMAGE_5(simd_type) \
__ ## simd_type image9;

#define DECLARE_IMAGE_7(simd_type) \
DECLARE_IMAGE_6(simd_type) \
__ ## simd_type imageA;

#define DECLARE_IMAGE_8(simd_type) \
DECLARE_IMAGE_7(simd_type) \
__ ## simd_type imageB;

/* Sums declare */
#define DECLARE_SUM_1(simd_type) \
__ ## simd_type sum0;

#define DECLARE_SUM_2(simd_type) \
DECLARE_SUM_1(simd_type) \
__ ## simd_type sum1;

#define DECLARE_SUM_4(simd_type) \
DECLARE_SUM_2(simd_type) \
__ ## simd_type sum2; \
__ ## simd_type sum3;

#define DECLARE_SUM_5(simd_type) \
DECLARE_SUM_4(simd_type) \
__ ## simd_type sum4;

#define DECLARE_SUM_6(simd_type) \
DECLARE_SUM_5(simd_type) \
__ ## simd_type sum5;

#define DECLARE_SUM_7(simd_type) \
DECLARE_SUM_6(simd_type) \
__ ## simd_type sum6;

#define DECLARE_SUM_8(simd_type) \
DECLARE_SUM_7(simd_type) \
__ ## simd_type sum7;

/* Sums load */
#define LOAD_SUM_1(simd_type) \
sum0 = _m ## simd_type ## _loadu_ps(output0);

#define LOAD_SUM_2(simd_type) \
LOAD_SUM_1(simd_type) \
sum1 = _m ## simd_type ## _loadu_ps(output1);

#define LOAD_SUM_4(simd_type) \
LOAD_SUM_2(simd_type) \
sum2 = _m ## simd_type ## _loadu_ps(output2); \
sum3 = _m ## simd_type ## _loadu_ps(output3);

#define LOAD_SUM_5(simd_type) \
LOAD_SUM_4(simd_type) \
sum4 = _m ## simd_type ## _loadu_ps(output4);

#define LOAD_SUM_6(simd_type) \
LOAD_SUM_5(simd_type) \
sum5 = _m ## simd_type ## _loadu_ps(output5);

#define LOAD_SUM_7(simd_type) \
LOAD_SUM_6(simd_type) \
sum6 = _m ## simd_type ## _loadu_ps(output6);

#define LOAD_SUM_8(simd_type) \
LOAD_SUM_7(simd_type) \
sum7 = _m ## simd_type ## _loadu_ps(output7);

/* Sums store */
#define STORE_SUM_1(simd_type) \
_m ## simd_type ## _storeu_ps(output0, sum0);

#define STORE_SUM_2(simd_type) \
STORE_SUM_1(simd_type) \
_m ## simd_type ## _storeu_ps(output1, sum1);

#define STORE_SUM_4(simd_type) \
STORE_SUM_2(simd_type) \
_m ## simd_type ## _storeu_ps(output2, sum2); \
_m ## simd_type ## _storeu_ps(output3, sum3);

#define STORE_SUM_5(simd_type) \
STORE_SUM_4(simd_type) \
_m ## simd_type ## _storeu_ps(output4, sum4);

#define STORE_SUM_6(simd_type) \
STORE_SUM_5(simd_type) \
_m ## simd_type ## _storeu_ps(output5, sum5);

#define STORE_SUM_7(simd_type) \
STORE_SUM_6(simd_type) \
_m ## simd_type ## _storeu_ps(output6, sum6);

#define STORE_SUM_8(simd_type) \
STORE_SUM_7(simd_type) \
_m ## simd_type ## _storeu_ps(output7, sum7);

/* Convolution */
#define CONVOLVE_1ROWS(simd_type) \
image0 = _m ## simd_type ## _loadu_ps(input0); \
image1 = _m ## simd_type ## _loadu_ps(input1); \
image2 = _m ## simd_type ## _loadu_ps(input2); \
image3 = _m ## simd_type ## _loadu_ps(input3); \
image4 = _m ## simd_type ## _loadu_ps(input4); \
\
sum0 = _m ## simd_type ## _add_ps(sum0, _m ## simd_type ## _mul_ps(weight0, image0)); \
sum0 = _m ## simd_type ## _add_ps(sum0, _m ## simd_type ## _mul_ps(weight1, image1)); \
sum0 = _m ## simd_type ## _add_ps(sum0, _m ## simd_type ## _mul_ps(weight2, image2)); \
sum0 = _m ## simd_type ## _add_ps(sum0, _m ## simd_type ## _mul_ps(weight3, image3)); \
sum0 = _m ## simd_type ## _add_ps(sum0, _m ## simd_type ## _mul_ps(weight4, image4));

#define CONVOLVE_2ROWS(simd_type) \
CONVOLVE_1ROWS(simd_type) \
image5 = _m ## simd_type ## _loadu_ps(input5); \
sum1 = _m ## simd_type ## _add_ps(sum1, _m ## simd_type ## _mul_ps(weight0, image1)); \
sum1 = _m ## simd_type ## _add_ps(sum1, _m ## simd_type ## _mul_ps(weight1, image2)); \
sum1 = _m ## simd_type ## _add_ps(sum1, _m ## simd_type ## _mul_ps(weight2, image3)); \
sum1 = _m ## simd_type ## _add_ps(sum1, _m ## simd_type ## _mul_ps(weight3, image4)); \
sum1 = _m ## simd_type ## _add_ps(sum1, _m ## simd_type ## _mul_ps(weight4, image5));

#define CONVOLVE_4ROWS(simd_type) \
CONVOLVE_2ROWS(simd_type) \
image6 = _m ## simd_type ## _loadu_ps(input6); \
sum2 = _m ## simd_type ## _add_ps(sum2, _m ## simd_type ## _mul_ps(weight0, image2)); \
sum2 = _m ## simd_type ## _add_ps(sum2, _m ## simd_type ## _mul_ps(weight1, image3)); \
sum2 = _m ## simd_type ## _add_ps(sum2, _m ## simd_type ## _mul_ps(weight2, image4)); \
sum2 = _m ## simd_type ## _add_ps(sum2, _m ## simd_type ## _mul_ps(weight3, image5)); \
sum2 = _m ## simd_type ## _add_ps(sum2, _m ## simd_type ## _mul_ps(weight4, image6)); \
\
image7 = _m ## simd_type ## _loadu_ps(input7); \
sum3 = _m ## simd_type ## _add_ps(sum3, _m ## simd_type ## _mul_ps(weight0, image3)); \
sum3 = _m ## simd_type ## _add_ps(sum3, _m ## simd_type ## _mul_ps(weight1, image4)); \
sum3 = _m ## simd_type ## _add_ps(sum3, _m ## simd_type ## _mul_ps(weight2, image5)); \
sum3 = _m ## simd_type ## _add_ps(sum3, _m ## simd_type ## _mul_ps(weight3, image6)); \
sum3 = _m ## simd_type ## _add_ps(sum3, _m ## simd_type ## _mul_ps(weight4, image7));

#define CONVOLVE_5ROWS(simd_type) \
CONVOLVE_4ROWS(simd_type) \
image8 = _m ## simd_type ## _loadu_ps(input8); \
sum4 = _m ## simd_type ## _add_ps(sum4, _m ## simd_type ## _mul_ps(weight0, image4)); \
sum4 = _m ## simd_type ## _add_ps(sum4, _m ## simd_type ## _mul_ps(weight1, image5)); \
sum4 = _m ## simd_type ## _add_ps(sum4, _m ## simd_type ## _mul_ps(weight2, image6)); \
sum4 = _m ## simd_type ## _add_ps(sum4, _m ## simd_type ## _mul_ps(weight3, image7)); \
sum4 = _m ## simd_type ## _add_ps(sum4, _m ## simd_type ## _mul_ps(weight4, image8));

#define CONVOLVE_6ROWS(simd_type) \
CONVOLVE_5ROWS(simd_type) \
image9 = _m ## simd_type ## _loadu_ps(input9); \
sum5 = _m ## simd_type ## _add_ps(sum5, _m ## simd_type ## _mul_ps(weight0, image5)); \
sum5 = _m ## simd_type ## _add_ps(sum5, _m ## simd_type ## _mul_ps(weight1, image6)); \
sum5 = _m ## simd_type ## _add_ps(sum5, _m ## simd_type ## _mul_ps(weight2, image7)); \
sum5 = _m ## simd_type ## _add_ps(sum5, _m ## simd_type ## _mul_ps(weight3, image8)); \
sum5 = _m ## simd_type ## _add_ps(sum5, _m ## simd_type ## _mul_ps(weight4, image9));

#define CONVOLVE_7ROWS(simd_type) \
CONVOLVE_6ROWS(simd_type) \
imageA = _m ## simd_type ## _loadu_ps(inputA); \
sum6 = _m ## simd_type ## _add_ps(sum6, _m ## simd_type ## _mul_ps(weight0, image6)); \
sum6 = _m ## simd_type ## _add_ps(sum6, _m ## simd_type ## _mul_ps(weight1, image7)); \
sum6 = _m ## simd_type ## _add_ps(sum6, _m ## simd_type ## _mul_ps(weight2, image8)); \
sum6 = _m ## simd_type ## _add_ps(sum6, _m ## simd_type ## _mul_ps(weight3, image9)); \
sum6 = _m ## simd_type ## _add_ps(sum6, _m ## simd_type ## _mul_ps(weight4, imageA));

#define CONVOLVE_8ROWS(simd_type) \
CONVOLVE_7ROWS(simd_type) \
imageB = _m ## simd_type ## _loadu_ps(inputB); \
sum7 = _m ## simd_type ## _add_ps(sum7, _m ## simd_type ## _mul_ps(weight0, image7)); \
sum7 = _m ## simd_type ## _add_ps(sum7, _m ## simd_type ## _mul_ps(weight1, image8)); \
sum7 = _m ## simd_type ## _add_ps(sum7, _m ## simd_type ## _mul_ps(weight2, image9)); \
sum7 = _m ## simd_type ## _add_ps(sum7, _m ## simd_type ## _mul_ps(weight3, imageA)); \
sum7 = _m ## simd_type ## _add_ps(sum7, _m ## simd_type ## _mul_ps(weight4, imageB));

/* Convolution MEGA macro */
#define DECLARE_SUMX(rows) DECLARE_SUM_ ## rows
#define LOAD_SUMX(rows) LOAD_SUM_ ## rows
#define DECLARE_INPUTX(rows) DECLARE_INPUT_ ## rows
#define DECLARE_IMAGEX(rows) DECLARE_IMAGE_ ## rows
#define CONVOLVEX(rows) CONVOLVE_ ## rows ## ROWS
#define INC_INPUTX(rows) INC_INPUT_ ## rows
#define STORE_SUMX(rows) STORE_SUM_ ## rows
#define INC_OUTPUTX(rows) INC_OUTPUT_ ## rows

#define CONVOLUTION_LOOP(rows, simd_type, simd_inst_prefex, simd_set, i) \
DECLARE_SUMX(rows)(simd_type) \
LOAD_SUMX(rows)(simd_inst_prefex) \
DECLARE_WEIGHTS(simd_type) \
DECLARE_INPUT_0(i) \
DECLARE_INPUTX(rows)() \
DECLARE_IMAGEX(rows)(simd_type) \
\
LOAD_WEIGHTS(0, simd_inst_prefex, simd_set) \
CONVOLVEX(rows)(simd_inst_prefex) \
INC_INPUTX(rows)() \
\
LOAD_WEIGHTS(1, simd_inst_prefex, simd_set) \
CONVOLVEX(rows)(simd_inst_prefex) \
INC_INPUTX(rows)() \
\
LOAD_WEIGHTS(2, simd_inst_prefex, simd_set) \
CONVOLVEX(rows)(simd_inst_prefex) \
INC_INPUTX(rows)() \
\
LOAD_WEIGHTS(3, simd_inst_prefex, simd_set) \
CONVOLVEX(rows)(simd_inst_prefex) \
INC_INPUTX(rows)() \
\
LOAD_WEIGHTS(4, simd_inst_prefex, simd_set) \
CONVOLVEX(rows)(simd_inst_prefex) \
\
STORE_SUMX(rows)(simd_inst_prefex) \
\
INC_OUTPUTX(rows)(sizeof(__ ## simd_type) / sizeof(float))


#define CONVOLVE_8COLS_XROWS(rows, i) \
{ \
CONVOLUTION_LOOP(rows, m256, m256, _set1_ps, i) \
}

#define CONVOLVE_4COLS_XROWS(rows, i) \
{ \
CONVOLUTION_LOOP(rows, m128, m, _set_ps1, i) \
}

#endif
