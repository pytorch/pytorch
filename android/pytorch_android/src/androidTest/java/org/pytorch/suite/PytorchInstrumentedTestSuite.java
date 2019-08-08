package org.pytorch.suite;

import org.pytorch.PytorchInstrumentedTests;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({PytorchInstrumentedTests.class})
public class PytorchInstrumentedTestSuite {
}
