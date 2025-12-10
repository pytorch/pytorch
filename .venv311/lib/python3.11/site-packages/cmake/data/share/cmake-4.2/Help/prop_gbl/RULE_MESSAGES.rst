RULE_MESSAGES
-------------

Specify whether to report a message for each make rule.

This property specifies whether Makefile generators should add a
progress message describing what each build rule does.  If the
property is not set the default is ON.  Set the property to OFF to
disable granular messages and report only as each target completes.
This is intended to allow scripted builds to avoid the build time cost
of detailed reports.  If a :variable:`CMAKE_RULE_MESSAGES` cache entry exists
its value initializes the value of this property.  Non-Makefile
generators currently ignore this property.
