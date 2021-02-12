cd test

# PYTORCH_TEST_WITH_ROCM=1 python test_spectral_ops.py --verbose

# python test_spectral_ops.py --verbose &>../scripts/amd/test_spectral.log

# FAILs
# python test_spectral_ops.py --verbose TestFFTCUDA.test_complex_stft_real_equiv_cuda_complex128 # AssertionError: False is not true : Tensors failed to compare as equal! Real parts failed to compare as equal! With rtol=1e-07 and atol=1e-07, found 2300 element(s) (out of 2300) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 6.338320066220198 (0.228597939843123 vs. 6.566918006063321), which occurred at index (18, 0).
# python test_spectral_ops.py --verbose TestFFTCUDA.test_complex_stft_roundtrip_cuda_float64 # AssertionError: False is not true : Tensors failed to compare as equal! With rtol=1e-07 and atol=1e-07, found 600 element(s) (out of 600) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 2.953264635629899 (0.6038054203149951 vs. -2.3494592153149036), which occurred at index 127.
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_ifft_rfft_irfft_cuda_float64 # AssertionError: False is not true : rfft and irfft
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_numpy_cuda_complex128 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing 13.996607399613103 and 14.117717213318706 gives a difference of 0.12110981370560303, but the allowed difference with rtol=1.3e-06 and atol=1e-05 is only 2.835303237731432e-05!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_numpy_cuda_complex64 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing -1.4140572547912598 and -2.904739187564701 gives a difference of 1.490681932773441, but the allowed difference with rtol=1.3e-06 and atol=0.0001 is only 0.00010377616094383411!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_numpy_cuda_float32 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing the real part 56.722625732421875 and 1.3659168034791946 gives a difference of 55.35670892894268, but the allowed difference with rtol=1.3e-06 and atol=0.0001 is only 0.00010177569184452296!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_numpy_cuda_float64 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing the real part 19.386301203284233 and 4.5661524249271706 gives a difference of 14.820148778357062, but the allowed difference with rtol=1.3e-06 and atol=1e-05 is only 1.5935998152405323e-05!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_round_trip_cuda_float32 # AssertionError: False is not true : Tensors failed to compare as equal! Real parts failed to compare as equal! With rtol=1.3e-06 and atol=1e-05, found 80 element(s) (out of 80) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 20.849252581596375 (19.01752471923828 vs. -1.8317278623580933), which occurred at index 21.
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_round_trip_cuda_float64 # AssertionError: False is not true : Tensors failed to compare as equal! Real parts failed to compare as equal! With rtol=1e-07 and atol=1e-07, found 80 element(s) (out of 80) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 19.260061657864185 (-19.416217897788812 vs. -0.1561562399246273), which occurred at index 33.
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_numpy_cuda_complex128 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing 0.11271160570600122 and 0.1380964338507653 gives a difference of 0.025384828144764088, but the allowed difference with rtol=1.3e-06 and atol=1e-05 is only 1.0179525364005995e-05!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_numpy_cuda_complex64 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing 0.06631094217300415 and -0.03396454837638885 gives a difference of 0.100275490549393, but the allowed difference with rtol=1.3e-06 and atol=0.0001 is only 0.00010004415391288932!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_numpy_cuda_float32 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing 0.07344067096710205 and 0.14093156158924103 gives a difference of 0.06749089062213898, but the allowed difference with rtol=1.3e-06 and atol=0.0001 is only 0.00010018321103006602!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_numpy_cuda_float64 # AssertionError: False is not true : Scalars failed to compare as equal! Comparing -0.1565018078567295 and -0.09710092023594495 gives a difference of 0.05940088762078456, but the allowed difference with rtol=1.3e-06 and atol=1e-05 is only 1.0126231196306729e-05!
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_round_trip_cuda_float32 # AssertionError: False is not true : Tensors failed to compare as equal! With rtol=1.3e-06 and atol=1e-05, found 20 element(s) (out of 20) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 4.629650115966797 (-3.9924628734588623 vs. 0.6371872425079346), which occurred at index (3, 4).
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_round_trip_cuda_float64 # AssertionError: False is not true : Tensors failed to compare as equal! With rtol=1e-07 and atol=1e-07, found 20 element(s) (out of 20) whose difference(s) exceeded the margin of error (including 0 nan comparisons). The greatest difference was 4.292929479694238 (-3.2269253570102054 vs. 1.0660041226840327), which occurred at index (3, 4).
# python test_spectral_ops.py --verbose TestFFTCUDA.test_stft_cuda_float64 # AssertionError: False is not true : stft comparison against librosa

# ERRORS
# python test_spectral_ops.py --verbose TestFFTCUDA.test_cufft_plan_cache_cuda_float64 # RuntimeError: cuFFT with HIP is not supported
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_backward_cuda_complex128 # RuntimeError: Jacobian mismatch for output 0 with respect to input 0,
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_backward_cuda_float64 # RuntimeError: Gradients failed to compare equal for grad output = 1j. Jacobian mismatch for output 0 with respect to input 0,
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_backward_cuda_complex128 # RuntimeError: Jacobian mismatch for output 0 with respect to input 0,
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fftn_backward_cuda_float64 #RuntimeError: Jacobian mismatch for output 0 with respect to input 0,

# CURRENT TEST
# python test_spectral_ops.py --verbose TestFFTCUDA.test_cufft_plan_cache_cuda_float64 # RuntimeError: cuFFT with HIP is not supported

# SUBTEST
# python test_spectral_ops.py --verbose TestFFTCUDA.test_fft_ifft_rfft_irfft_cuda_float64 # AssertionError: False is not true : rfft and irfft

python test_spectral_ops.py --verbose TestFFTCUDA.test_batch_istft_cuda
