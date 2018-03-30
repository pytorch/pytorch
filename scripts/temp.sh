find ../caffe2 -name "*.py" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.h" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.cc" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.cpp" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.cu" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.mm" -exec ./remove_apache_header.sh {} \;
find ../caffe2 -name "*.m" -exec ./remove_apache_header.sh {} \;
