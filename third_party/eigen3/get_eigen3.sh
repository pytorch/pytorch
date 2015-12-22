# Run get_eigen3.sh to update the protocol buffer to the most recent version.
# Current Eigen version:
#    Eigen 3.3 beta 1

rm -rf Eigen
rm -rf unsupported

wget http://bitbucket.org/eigen/eigen/get/3.3-beta1.tar.bz2
bunzip2 *.tar.bz2
tar xf *.tar
mv eigen-eigen-*/Eigen .
mv eigen-eigen-*/unsupported .
rm -rf eigen-eigen*
rm *.tar
