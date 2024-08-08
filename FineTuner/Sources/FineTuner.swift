//
//  File.swift
//  
//
//  Created by Peter Schuette on 7/26/24.
//

import Foundation
import ArgumentParser
import CoreML
import CoreVideo

@main
struct FineTuner: AsyncParsableCommand {
    @Option(
        name: [.customShort("m"), .customLong("model")],
        help: "path to .coreml model",
        transform: URL.init(fileURLWithPath:)
    )
    var modelPath: URL
    
    @Option(
        name: [.customShort("t"), .customLong("trainingData")],
        help: "path to training dataset",
        transform: URL.init(fileURLWithPath:)
    )
    var datasetPath: URL
    
    mutating func run() async throws {
        
        let compiledModelPath = try await CoreML.MLModel.compileModel(at: modelPath)

        print("Tuning model: \(modelPath.absoluteString)")
        print("With data: \(datasetPath)")
        print(compiledModelPath.description)
        
        let tunedModel = try await train(compiledModelPath, from: datasetPath)
        
        print("tuned: ")
    }
    
//    private func pixelBuffer(for imagePath: URL) -> CVPixelBuffer? {
//        var pixelBuffer: CVPixelBuffer?
//
//        let attributes: [CFString: Any] = [
//            kCVPixelBufferCGImageCompatibilityKey: true,
//            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
//            kCVPixelBufferWidthKey: 28,
//            kCVPixelBufferHeightKey: 28,
//            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_16Gray,
//        ]
//        
//        let status = CVPixelBufferCreate(
//            kCFAllocatorDefault,
//            28,
//            28,
//            kCVPixelFormatType_16Gray,
//            attributes as CFDictionary,
//            &pixelBuffer
//        )
//        
//        guard status == kCVReturnSuccess else {
//            print("Error: Could not create pixel buffer")
//            return nil
//        }
//        return pixelBuffer
//    }
    
    private func train(_ compiledModelPath: URL, from datasetPath: URL) async throws -> MLModel {
        let compiledModel = try CoreML.MLModel(contentsOf: compiledModelPath)

        let directories = try FileManager.default.contentsOfDirectory(atPath: datasetPath.path()).filter {
            var isDirectory: ObjCBool = false
            let classDataPath = datasetPath.appending(component: $0).path()
            return FileManager.default.fileExists(
                atPath: classDataPath,
                isDirectory: &isDirectory
            ) && isDirectory.boolValue
        }
        
        for trainingClass in directories {
            // Train one digit at a time to avoid system limits
            let subsetDirectory = datasetPath.appending(component: trainingClass)
            guard let values = try? FileManager.default.contentsOfDirectory(atPath: subsetDirectory.path()) else {
                print("failed to read contents - should throw")
                break
            }
            
            let featureValues = values.compactMap { file in
                let imagePath = subsetDirectory.appending(component: file)
                // Should we validate the image here?
                return try? MLFeatureValue(imageAt: imagePath, pixelsWide: 28, pixelsHigh: 28, pixelFormatType: kCVPixelFormatType_16Gray)
            }
            // TODO make this batches of fonts so we are training all class values at the same time and can control the maximum size more easily
            let batchProvider = try MLArrayBatchProvider(dictionary: [trainingClass: featureValues])
            
            let model = try await withCheckedThrowingContinuation { continuation in
                do {
                    let task = try MLUpdateTask(
                        forModelAt: compiledModelPath,
                        trainingData: batchProvider
                    ) { context in
                        continuation.resume(with: .success(context.model))
                    }
                    
                    task.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        
        return compiledModel
    }
}

extension FineTuner {
    enum Error: Swift.Error {
        case missingModel
        case missingDataset
        case missingPixelBuffer
    }
}
