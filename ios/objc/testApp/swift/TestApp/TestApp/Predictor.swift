import UIKit
import PytorchObjC

struct InferenceResult {
    let score: Float32
    let label: String
}

class Predictor {
    var module: TorchModule?
    var labels: [String]?
    
    init(modelPath: String) {
        self.module = TorchModule.loadTorchscriptModel(modelPath)
        self.labels = loadLabel()
    }
    
    func predict(image:UIImage, completion:(_ results:[InferenceResult])->Void){
        guard var data:[Float32] = image.normalizedBuffer() else{
            completion([])
            return
        }
        let imageTensor = TorchTensor.new(with: .float, size: [1,3,224,224], data: UnsafeMutablePointer(&data))
        guard let inputTensor = imageTensor else {
            completion([])
            return
        }
        let imageIValue = TorchIValue.new(with: inputTensor)
        let outputTensor = self.module?.forward([imageIValue])?.toTensor()
        
        guard let resultTensor = outputTensor else {
            completion([])
            return
        }
        let topKResuls = self.getTopN(results: resultTensor, count: 5)
        completion(topKResuls);
    }
    
    private func getTopN(results: TorchTensor, count: Int) -> [InferenceResult] {
        guard let labels = self.labels else {
            return []
        }
        let resultCount: UInt = results.size[1].uintValue
        var scores: [Float32] = []
        for index: UInt in 0..<resultCount {
            if let score = results[0]?[index]?.item()?.floatValue {
                scores.append(score)
            }
        }
        let zippedResults = zip(labels.indices,scores)
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(count)
        return sortedResults.map({ result in InferenceResult(score: result.1, label: labels[result.0])})
    }
    
    private func loadLabel() -> [String]{
        if let filePath = Bundle.main.path(forResource: "synset_words", ofType: "txt"),
            let labels = try? String(contentsOfFile:  filePath){
            return labels.components(separatedBy: .newlines);
        }else {
            fatalError("Label file was not found.")
        }
    }
}
