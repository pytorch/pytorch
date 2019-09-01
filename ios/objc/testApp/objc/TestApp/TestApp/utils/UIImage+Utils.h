#import <UIKit/UIKit.h>
#import <memory>

NS_ASSUME_NONNULL_BEGIN

@interface UIImage (Utils)

- (UIImage* )resize:(CGSize)sz;
- (float* )normalizedBuffer;

@end

NS_ASSUME_NONNULL_END
