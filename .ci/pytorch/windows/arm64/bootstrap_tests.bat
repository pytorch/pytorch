:: change to source directory
cd %PYTORCH_ROOT%

:: activate visual studio
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: add APL to PATH for runtime dependencies
if exist "%DEPENDENCIES_DIR%\armpl_24.10\bin" set PATH=%DEPENDENCIES_DIR%\armpl_24.10\bin;%PATH%

:: ------------------------------------------------------------------
:: Ensure MSVC ARM64 runtime DLLs are discoverable at runtime
:: IMPORTANT: use delayed expansion to avoid PATH parsing errors
:: ------------------------------------------------------------------
setlocal EnableDelayedExpansion

for /d %%D in (
  "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Redist\MSVC\*\arm64\Microsoft.VC143.CRT"
) do (
  if exist "%%D\vcruntime140.dll" (
    set "PATH=%%D;!PATH!"
  )
)

endlocal & set "PATH=%PATH%"

:: install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest numpy protobuf expecttest hypothesis

:: find file name for pytorch wheel
for /f "delims=" %%f in ('dir /b "%PYTORCH_FINAL_PACKAGE_DIR%" ^| findstr "torch-"') do set "TORCH_WHEEL_FILENAME=%PYTORCH_FINAL_PACKAGE_DIR%\%%f"

pip install %TORCH_WHEEL_FILENAME%