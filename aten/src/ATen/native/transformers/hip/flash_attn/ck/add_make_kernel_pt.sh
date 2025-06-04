#!/bin/bash

# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_list.txt>"
    exit 1
fi

# Assign the input file to a variable
file_list=$1

# Check if the file exists
if [ ! -f "$file_list" ]; then
    echo "Error: File '$file_list' not found!"
    exit 1
fi

# Loop through each line in the file list
while IFS= read -r file; do
    # Check if the file exists in the current directory
    if [ -f "$file" ]; then
        # Use sed to replace "make_kernel" with "make_kernel_pt" in place
        sed -i 's/make_kernel/make_kernel_pt/g' "$file"
        echo "Updated: $file"
    else
        echo "Skipping: $file (not found)"
    fi
done < "$file_list"

echo "Replacement completed."
