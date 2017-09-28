find ../caffe2 -name "*.h" -exec ./add_apache_header.sh {} \;
find ../caffe2 -name "*.cc" -exec ./add_apache_header.sh {} \;
find ../caffe2 -name "*.cpp" -exec ./add_apache_header.sh {} \;
find ../caffe2 -name "*.cu" -exec ./add_apache_header.sh {} \;
find ../caffe2 -name "*.mm" -exec ./add_apache_header.sh {} \;
find ../caffe2 -name "*.m" -exec ./add_apache_header.sh {} \;
