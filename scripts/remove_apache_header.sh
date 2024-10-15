if [[ "$1" == *.py ]]; then
  apache_header="apache_python.txt"
else
  apache_header="apache_header.txt"
fi
apache_lines=$(wc -l < "${apache_header}")
apache_md5=$(cat "${apache_header}" | md5)
header_md5=$(head -n ${apache_lines} $1 | md5)
if [ "${header_md5}" == "${apache_md5}" ]; then
  keep_lines=$(($(wc -l < $1) - ${apache_lines}))
  tail -n ${keep_lines} $1 > _remove_apache_header.txt
  mv _remove_apache_header.txt $1
fi
