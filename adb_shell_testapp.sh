#!/bin/bash
adb shell am force-stop org.pytorch.testapp

adb shell am start -n org.pytorch.testapp.mbq/org.pytorch.testapp.MainActivity
