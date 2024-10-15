#import "ViewController.h"


#ifdef BUILD_LITE_INTERPRETER
#import "Benchmark.h"
#endif

@interface ViewController ()
@property(nonatomic, strong) UITextView* textView;
@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

#ifdef BUILD_LITE_INTERPRETER
  self.textView = [[UITextView alloc] initWithFrame:self.view.bounds];
  self.textView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:self.textView];

  NSData* configData = [NSData dataWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"config" ofType:@"json"]];
  if (!configData) {
    NSLog(@"Config.json not found!");
    return;
  }

  NSError* err;
  NSDictionary* config = [NSJSONSerialization JSONObjectWithData:configData options:NSJSONReadingAllowFragments error:&err];
  if (err) {
    NSLog(@"Parse config.json failed!");
    return;
  }
// NB: When running tests on device, we need an empty app to launch the tests
#ifdef RUN_BENCHMARK
  [Benchmark setup:config];
  [self runBenchmark];
#endif
#endif
}

#ifdef BUILD_LITE_INTERPRETER
- (void)runBenchmark {
  self.textView.text = @"Start benchmarking...\n";
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSString* text = [Benchmark run];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.textView.text = [self.textView.text stringByAppendingString:text];
    });
  });
}
#endif

@end
