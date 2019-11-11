#import "ViewController.h"
#import <torch/script.h>
#import "Benchmark.h"

@interface ViewController ()
@property(weak, nonatomic) IBOutlet UITextView* textView;

@end

@implementation ViewController {
}

- (void)viewDidLoad {
  [super viewDidLoad];

  NSError* err;
  NSData* configData = [NSData
      dataWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"config" ofType:@"json"]];
  if (!configData) {
    NSLog(@"Config.json not found!");
    return;
  }
  NSDictionary* config = [NSJSONSerialization JSONObjectWithData:configData
                                                         options:NSJSONReadingAllowFragments
                                                           error:&err];

  if (err) {
    NSLog(@"Parse config.json failed!");
    return;
  }
  [Benchmark setup:config];
  [self runBenchmark];
}

- (void)runBenchmark {
  self.textView.text = @"Start benchmarking...\n";
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSString* text = [Benchmark run];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.textView.text = [self.textView.text stringByAppendingString:text];
    });
  });
}

- (IBAction)reRun:(id)sender {
  self.textView.text = @"";
  dispatch_async(dispatch_get_main_queue(), ^{
    [self runBenchmark];
  });
}

@end
