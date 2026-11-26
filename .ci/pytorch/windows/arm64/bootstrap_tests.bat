:: change to source directory
cd %PYTORCH_ROOT%

:: activate visual studio
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -find VC\Auxiliary\Build\vcvarsall.bat`) do call "%%i" arm64
where cl.exe

:: create virtual environment
if not defined DESIRED_PYTHON_EXE set DESIRED_PYTHON_EXE=python
"%DESIRED_PYTHON_EXE%" -m venv .venv
if errorlevel 1 exit /b 1
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: install dependencies
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install --only-binary=numpy -c "%~dp0requirements-build-constraints.txt" -r requirements.txt pytest protobuf expecttest hypothesis
if errorlevel 1 exit /b 1

:: find file name for pytorch wheel
for /f "delims=" %%f in ('dir /b "%PYTORCH_FINAL_PACKAGE_DIR%" ^| findstr "torch-"') do set "TORCH_WHEEL_FILENAME=%PYTORCH_FINAL_PACKAGE_DIR%\%%f"

python -m pip install -c "%~dp0requirements-build-constraints.txt" "%TORCH_WHEEL_FILENAME%"