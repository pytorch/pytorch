mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  if "%BUILD_ENVIRONMENT%"=="" (
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/LLVM-10.0.0-win64.exe --output %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/llvm-10.0.0-with-D77920.7z --output %TMP_DIR_WIN%\llvm-10.0.0-with-D77920.7z
  ) else (
    aws s3 cp s3://ossci-windows/LLVM-10.0.0-win64.exe %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe --quiet
    aws s3 cp s3://ossci-windows/llvm-10.0.0-with-D77920.7z %TMP_DIR_WIN%\llvm-10.0.0-with-D77920.7z --quiet
  )
  7z x -aoa %TMP_DIR_WIN%\LLVM-10.0.0-win64.exe -o"C:\Program Files\LLVM"
  7z x -aoa %TMP_DIR_WIN%\llvm-10.0.0-with-D77920.7z -o"C:\Program Files\LLVM"
)
