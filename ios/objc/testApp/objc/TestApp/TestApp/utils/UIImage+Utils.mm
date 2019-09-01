#import "UIImage+Utils.h"

@implementation UIImage (Utils)

- (UIImage* )resize:(CGSize)sz {
    if(CGSizeEqualToSize(self.size, sz)){
        return self;
    }
    UIGraphicsBeginImageContextWithOptions(sz, NO, 1);
    //decode the image to RGBA
    [self drawInRect:(CGRect){{0,0},{sz}}];
    UIImage* resizedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return resizedImage;
}

- (float* )normalizedBuffer {
    CGImageRef inputCGImage = self.CGImage;
    NSUInteger width = CGImageGetWidth(inputCGImage);
    NSUInteger height = CGImageGetHeight(inputCGImage);
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    uint8_t * rawPixels = (uint8_t *) calloc(height * width * bytesPerPixel, sizeof(uint8_t));
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(rawPixels, width, height, bitsPerComponent, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), inputCGImage);
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    float* normalizedBuffer = (float *) calloc(height * width * 3, sizeof(float));
    //normalize the pixel buffer
    //see https://pytorch.org/hub/pytorch_vision_resnet/ for more detail
    for(NSUInteger i=0; i<height * width; ++i){
        normalizedBuffer[i]              = (rawPixels[i*4+0] / 255.0 - 0.485) / 0.229; //R
        normalizedBuffer[width*height+i] = (rawPixels[i*4+1] / 255.0 - 0.456) / 0.224; //G
        normalizedBuffer[width*height*2] = (rawPixels[i*4+2] / 255.0 - 0.406) / 0.225; //B
    }
    free(rawPixels);
    return normalizedBuffer;
}

@end
