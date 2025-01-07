#!/bin/bash
set -ex

file_renaming_txt="rename_ck_autogen_files.output.txt"
rm -rf $file_renaming_txt
for file in `ls fmha_*wd*hip`; do
  sha1=$(sha1sum $file | cut -d' ' -f1)
  new_file="fmha_ck_autogen_${sha1}.hip"
  mv $file $new_file
  echo "$file -> $new_file" >> $file_renaming_txt
done
