(
  echo #!/bin/bash
  echo export INSTALLER_DIR="$SCRIPT_HELPERS_DIR"/installation-helpers
  echo source "$INSTALLER_DIR"/activate_miniconda3.sh
  echo echo "set > %INSTALLER_DIR%\tmp.txt" ^> $INSTALLER_DIR/tmp.bat
  echo "$INSTALLER_DIR/tmp.bat"
) > %INSTALLER_DIR%\tmp.sh
cat %INSTALLER_DIR%\tmp.sh
bash %INSTALLER_DIR%\tmp.sh
if errorlevel 1 exit /b
if not errorlevel 0 exit /b

rm %INSTALLER_DIR%\tmp.sh
rm %INSTALLER_DIR%\tmp.bat

FOR /F "tokens=*" %%i in ('type "%INSTALLER_DIR%\tmp.txt"') do SET %%i

rm %INSTALLER_DIR%\tmp.txt
