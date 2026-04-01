// profile-ignored.slang

// Cases where a `-profile` option gets ignored
// because it doesn't apply to any target

// Case 1: multiple (conflicting) profiles, so we can't infer a single stage
//
//DIAGNOSTIC_TEST:SIMPLE:-profile sm_5_0 -profile glsl_450


// Case 2: a `-profile` option before any `-target`, possibly because
// the user is specifying things in the wrong order.
//
//DIAGNOSTIC_TEST:SIMPLE:-profile sm_5_0 -target dxbc -profile glsl_450 -target spirv 
