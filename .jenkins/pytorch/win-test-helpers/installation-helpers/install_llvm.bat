mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  curl --retry 3 -kL "https://www.dropbox.com/s/b2m742ckj6j0u6k/llvm-14.0.0.7z?dl=1" --output %TMP_DIR_WIN%\llvm-14.0.0.7z
  7z x -aoa %TMP_DIR_WIN%\llvm-14.0.0.7z -o"C:\Program Files\LLVM"
)
