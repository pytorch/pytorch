package org.pytorch.suite;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.pytorch.PytorchInstrumentedTests;

@RunWith(Suite.class)
@Suite.SuiteClasses({PytorchInstrumentedTests.class})
public class PytorchInstrumentedTestSuite {}
