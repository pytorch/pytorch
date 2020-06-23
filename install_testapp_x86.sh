#!/bin/bash

gradle -p android test_app:installMbqLocalBaseDebug -PABI_FILTERS=x86
