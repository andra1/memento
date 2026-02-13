@preconcurrency import AVFoundation
import Foundation

enum ClipExtractionError: Error {
    case cannotCreateExportSession
    case exportFailed(Error?)
}

@MainActor
final class ClipExtractionService {
    func extractClip(
        from sourceURL: URL,
        to outputURL: URL,
        startTime: TimeInterval,
        endTime: TimeInterval
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            throw ClipExtractionError.cannotCreateExportSession
        }

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .m4a

        let safeStart = max(0, startTime)
        let safeDuration = max(0.1, endTime - safeStart)
        exportSession.timeRange = CMTimeRange(
            start: CMTime(seconds: safeStart, preferredTimescale: 600),
            duration: CMTime(seconds: safeDuration, preferredTimescale: 600)
        )

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            exportSession.exportAsynchronously {
                switch exportSession.status {
                case .completed:
                    continuation.resume(returning: ())
                case .failed, .cancelled:
                    continuation.resume(throwing: ClipExtractionError.exportFailed(exportSession.error))
                default:
                    continuation.resume(throwing: ClipExtractionError.exportFailed(exportSession.error))
                }
            }
        }

        return outputURL
    }
}
