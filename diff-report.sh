for new_file in ~/logs/mm/new_*; do
    base_name=$(basename "$new_file" | sed 's/^new_//')
    old_file="~/logs/mm/old_$base_name"
    if [ -f "$old_file" ]; then
        echo "$base_name"
        python diff_out.py "$new_file" "$old_file"
    fi
done
