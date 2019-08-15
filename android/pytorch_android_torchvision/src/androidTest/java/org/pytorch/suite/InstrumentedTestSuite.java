package org.pytorch.suite;

import org.pytorch.TorchVisionInstrumentedTests;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({TorchVisionInstrumentedTests.class})
public class InstrumentedTestSuite {
}
