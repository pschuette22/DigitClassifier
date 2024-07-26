//
//  File.swift
//  
//
//  Created by Peter Schuette on 7/26/24.
//

import Foundation
import ArgumentParser
import CoreML

@main
struct FineTuner: AsyncParsableCommand {
    @Option(
        name: [.customShort("m"), .customLong("model")],
        help: "path to .coreml model",
        transform: URL.init(fileURLWithPath:)
    )
    var modelPath: URL
    
    @Option(
        name: [.customShort("d"), .customLong("dataset")],
        help: "path to dataset",
        transform: URL.init(fileURLWithPath:)
    )
    var datasetPath: URL
    
    mutating func run() async throws {
        let model = try await CoreML.MLModel.compileModel(at: modelPath)

        print("Tuning model: \(modelPath.absoluteString)")
        print("With data: \(datasetPath)")
        print(model.description)
    }
}

extension FineTuner {
    enum Error: Swift.Error {
        case missingModel
        case missingDataset
    }
}
