BASEDIR=$(dirname "$0")
(< $BASEDIR/generated_dirs.txt xargs -i find {} -type f) | xargs git add -f
