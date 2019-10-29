#import "ViewController.h"
#import <torch/script.h>
#import "Benchmark.h"

@interface ViewController ()
@property(weak, nonatomic) IBOutlet UITextView* textView;

@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  NSError* err;
  NSData* configData = [NSData
      dataWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"config" ofType:@"json"]];
  NSDictionary* config = [NSJSONSerialization JSONObjectWithData:configData
                                                         options:NSJSONReadingAllowFragments
                                                           error:&err];
  if (err) {
    NSLog(@"Parse config.json failed!");
    return;
  }

  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    if ([Benchmark setup:config]) {
      NSString* text = [Benchmark run];
      dispatch_async(dispatch_get_main_queue(), ^{
        self.textView.text = text;
      });
    } else {
      NSLog(@"Setup benchmark config failed!");
    }
  });
}

@end
