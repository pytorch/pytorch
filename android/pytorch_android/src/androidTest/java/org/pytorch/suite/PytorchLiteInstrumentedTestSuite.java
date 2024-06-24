package org.pytorch.suite;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.pytorch.PytorchLiteInstrumentedTests;

@RunWith(Suite.class)
@Suite.SuiteClasses({PytorchLiteInstrumentedTests.class})
public class PytorchLiteInstrumentedTestSuite {}
