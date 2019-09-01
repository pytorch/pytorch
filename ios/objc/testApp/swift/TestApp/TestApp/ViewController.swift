import UIKit

class ViewController: UIViewController {
    var predictor: Predictor?
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if let modelPath = Bundle.main.path(forResource: "resnet18", ofType: "pt") {
            self.predictor = Predictor(modelPath: modelPath)
        }
        let image = UIImage(named: "wolf_400x400.jpg")!
        let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
        
        predictor?.predict(image: resizedImage, completion: { results in
            for result in results {
                print("-[score]: \(result.score), -[label]: \(result.label)")
            }
        })
    }
}

