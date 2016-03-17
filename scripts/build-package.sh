#!/bin/bash

# The envirinment set by create-ubuntu-environmens sscript is trusty, so we should match it.
# Look at debian/gbp.conf for default optiuons sent to gbp.
rm -r debian/tmp debian/*.log
git commit -a -m 'Preparing for pbuild'
# && git push
git clean -f

gbp buildpackage --git-pbuilder --git-dist=nvidia-7-5 $1 $2 $3 $4 $5
