mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  :check_sccache
  %TMP_DIR_WIN%\bin\sccache.exe --show-stats || (
    taskkill /im sccache.exe /f /t || ver > nul
    del %TMP_DIR_WIN%\bin\sccache.exe || ver > nul
    del %TMP_DIR_WIN%\bin\sccache-cl.exe || ver > nul
    if "%BUILD_ENVIRONMENT%"=="" (
      echo "TEST CI"
      curl --retry 3 --retry-all-errors -k https://ossci-windows.s3.amazonaws.com/sccache-v0.7.4.exe  --output %TMP_DIR_WIN%\bin\sccache.exe 
          ) else (
      aws s3 cp s3://ossci-windows.s3.amazonaws.com/sccache-v0.7.4.exe %TMP_DIR_WIN%\bin\sccache.exe
    )
    goto :check_sccache
  )
)
