#import "ViewController.h"
#import <torch/script.h>
#import "Benchmark.h"

@interface ViewController ()
@property(weak, nonatomic) IBOutlet UITextView* textView;

@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"model" ofType:@"pt"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
      NSString* text = [Benchmark benchmarkWithModel:modelPath];
      dispatch_async(dispatch_get_main_queue(), ^{
        self.textView.text = text;
      });
    } else {
      NSLog(@"model doesn't exist!");
    }
  });
}

@end
