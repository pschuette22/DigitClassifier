//
//  File.swift
//  FineTuner
//
//  Created by Peter Schuette on 7/29/24.
//

import Foundation
import XCTest
@testable import FineTuner

final class FineTunerTests: XCTestCase {
    func testFineTuner() async throws {
        let workspace = FileManager.default.homeDirectoryForCurrentUser.appending(path: "Workspace")
        let modelPath = workspace.appending(path: "DigitClassifier/product/UpdatableMNISTClassifier.mlpackage/Data/com.apple.CoreML/model.mlmodel")
        let datasetPath = workspace.appending(path: "DigitClassifier/dataset/fonts/train")
        var command = try FineTuner.parse([
            "-m",
            modelPath.path(),
            "-t",
            datasetPath.path()
        ])
        
        let _ = try await command.run()
    }
}
