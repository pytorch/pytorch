:: change to source directory
cd %PYTORCH_ROOT%

:: activate visual studio
call "%DEPENDENCIES_DIR%\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
where cl.exe

:: create virtual environment
python -m venv .venv
echo * > .venv\.gitignore
call .\.venv\Scripts\activate
where python

:: install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest numpy

:: find file name for pytorch wheel
for /f "delims=" %%f in ('dir /b "%PYTORCH_FINAL_PACKAGE_DIR%" ^| findstr "torch-"') do set "TORCH_WHEEL_FILENAME=%PYTORCH_FINAL_PACKAGE_DIR%\%%f"

pip install %TORCH_WHEEL_FILENAME%