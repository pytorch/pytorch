#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ImagePredictor : NSObject

- (instancetype)initWithModelPath:(NSString* )modelPath;
- (void)predict:(UIImage* )image
     Completion:(void(^__nullable)(NSArray<NSDictionary* >* sortedResults)) completion;

@end

NS_ASSUME_NONNULL_END
