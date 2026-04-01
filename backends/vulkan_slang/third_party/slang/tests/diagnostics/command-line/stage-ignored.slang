// stage-ignored.slang

// Cases where a `-stage` option gets ignored
// because it doesn't apply to any entry point.

// A `-stage` option before any `-entry`, possibly because
// the user is specifying things in the wrong order.
//
//DIAGNOSTIC_TEST:SIMPLE:-stage vertex -entry vsMain -stage fragment -entry psMain
