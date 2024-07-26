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
    @Option(name: [.customShort("m")], help: "path to .coreml model")
    var model: String
    private var modelPath: URL? {
        URL(string: model, relativeTo: .currentDirectory())
    }
    
    @Option(help: "path to dataset")
    var dataset: String
    private var datasetPath: URL? {
        URL(string: dataset, relativeTo: .currentDirectory())
    }
    
    mutating func run() async throws {
        guard let modelPath else { throw Error.missingModel }
        guard let datasetPath else { throw Error.missingDataset }
        
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
