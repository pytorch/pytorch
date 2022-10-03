#import "ViewController.h"

#import "Benchmark.h"

@interface ViewController ()
@property(nonatomic, strong) UITextView* textView;
@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

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

@end
