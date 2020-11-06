#!/bin/bash

#for file in *.h
#do
#    vim +':1,1s/^\n//eg' +':wq' "$file"
#done

#for file in *.cpp
#do
#    vim +':1,1s/^\n//eg' +':wq' "$file"
#done

vim +':1,1s/^\n//e' +':wq' ./kernel_resource*
