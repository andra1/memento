import AVFoundation
import CoreMedia
import Foundation
import SoundAnalysis

final class LaughterDetectionService: NSObject, SNResultsObserving {
    var onLaughterDetected: ((TimeInterval, Double) -> Void)?

    private var analyzer: SNAudioStreamAnalyzer?
    private var isConfigured = false

    func analyze(buffer: AVAudioPCMBuffer, when: AVAudioTime) {
        if !isConfigured {
            configureAnalyzer(with: buffer.format)
        }

        guard let analyzer else { return }
        analyzer.analyze(buffer, atAudioFramePosition: when.sampleTime)
    }

    private func configureAnalyzer(with format: AVAudioFormat) {
        analyzer = SNAudioStreamAnalyzer(format: format)
        isConfigured = true

        do {
            let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
            try analyzer?.add(request, withObserver: self)
        } catch {
            return
        }
    }

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let classificationResult = result as? SNClassificationResult else { return }
        guard let topClassification = classificationResult.classifications.first else { return }

        let identifier = topClassification.identifier.lowercased()
        let confidence = Double(topClassification.confidence)

        guard identifier.contains("laughter"), confidence >= 0.45 else { return }

        let seconds = classificationResult.timeRange.start.seconds
        onLaughterDetected?(seconds, confidence)
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {
    }

    func requestDidComplete(_ request: SNRequest) {
    }
}
