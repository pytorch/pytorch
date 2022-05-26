#!/bin/sh
#
# An example hook script to verify what is about to be committed.
# Called by git-commit with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.

# This is slightly modified from Andrew Morton's Perfect Patch.
# Lines you introduce should not have trailing whitespace.
# Also check for an indentation that has SP before a TAB.

if git-rev-parse --verify HEAD 2>/dev/null
then
	git-diff-index -p -M --cached HEAD
else
	# NEEDSWORK: we should produce a diff with an empty tree here
	# if we want to do the same verification for the initial import.
	:
fi |
perl -e '
    my $found_bad = 0;
    my $filename;
    my $reported_filename = "";
    my $lineno;
    sub bad_line {
	my ($why, $line) = @_;
	if (!$found_bad) {
	    print STDERR "*\n";
	    print STDERR "* You have some suspicious patch lines:\n";
	    print STDERR "*\n";
	    $found_bad = 1;
	}
	if ($reported_filename ne $filename) {
	    print STDERR "* In $filename\n";
	    $reported_filename = $filename;
	}
	print STDERR "* $why (line $lineno)\n";
	print STDERR "$filename:$lineno:$line\n";
    }
    while (<>) {
	if (m|^diff --git a/(.*) b/\1$|) {
	    $filename = $1;
	    next;
	}
	if (/^@@ -\S+ \+(\d+)/) {
	    $lineno = $1 - 1;
	    next;
	}
	if (/^ /) {
	    $lineno++;
	    next;
	}
	if (s/^\+//) {
	    $lineno++;
	    chomp;
	    if (/\s$/) {
		bad_line("trailing whitespace", $_);
	    }
	    if (/^\s* 	/) {
		bad_line("indent SP followed by a TAB", $_);
	    }
	}
    }
    exit($found_bad);
'
