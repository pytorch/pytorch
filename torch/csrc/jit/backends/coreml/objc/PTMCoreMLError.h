#import <Foundation/Foundation.h>

extern NSString* const kPTMCoreMLErrorDomain = @"kPTMCoreMLErrorDomain";

enum PTMCoreMLErrorCode {
  PTMCoreMLErrorCodeFailedSave,
  PTMCoreMLErrorCodeOSVersion,
  PTMCoreMLErrorCodeUnknown,
};
