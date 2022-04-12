$CMATH_DOWNLOAD_LINK = "https://raw.githubusercontent.com/microsoft/STL/12c684bba78f9b032050526abdebf14f58ca26a3/stl/inc/cmath"
$VC14_28_INSTALL_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910\include"

curl.exe --retry 3 -kL $CMATH_DOWNLOAD_LINK --output "$home\cmath"
Move-Item -Path "$home\cmath" -Destination "$VC14_28_INSTALL_PATH" -Force
