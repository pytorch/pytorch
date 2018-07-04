set -vex

if [ -z "$CAFFE2_BINARY_DIR" ] ; then
  if [ -z "$1" ] ; then
    CAFFE2_BINARY_DIR=.
  else
    CAFFE2_BINARY_DIR=$1
  fi
fi

files=($(find "$CAFFE2_BINARY_DIR" -type f -name "*_test"))
for test_binary in "${files[@]}";
do
  test_binary_base=$(basename $test_binary)
  if [[ $test_binary_base == gl* ]];then
    echo Running $test_binary_base
    adb push $test_binary "/data/local/tmp/$test_binary_base"
    adb shell "GLOG_logtostderr=1 /data/local/tmp/$test_binary_base" 
  fi
done

echo All tests passed.
