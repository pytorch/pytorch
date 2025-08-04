
set CMAKE_PREFIX_PATH=C:\Users\binba\local\pytorch
set PATH=C:\Users\binba\local\pytorch\torch\lib;%PATH%
set PYTHONPATH=C:\Users\binba\local\pytorch

cd model\data\aotinductor\model
rmdir /s /q build
mkdir build
cd build
cmake ..
cmake --build . --config Release

copy /Y *.cubin ..
copy /Y Release\* ..
cd ..
rmdir /s /q build
rm model_consts.cpp

cd ..\..\..\..\
python -c "import torch; from torch.export.pt2_archive._package import package_pt2; package_pt2('model.pt2', aoti_files=['model\\data\\aotinductor\\model'])"
