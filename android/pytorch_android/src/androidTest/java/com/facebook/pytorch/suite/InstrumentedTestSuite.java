package com.facebook.pytorch.suite;

import com.facebook.pytorch.InstrumentedTests;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({InstrumentedTests.class})
public class InstrumentedTestSuite {
}
