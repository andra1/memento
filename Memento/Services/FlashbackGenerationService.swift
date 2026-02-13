import Foundation
import SwiftData

@MainActor
protocol ClipExtractionServicing {
    func extractClip(
        from sourceURL: URL,
        to outputURL: URL,
        startTime: TimeInterval,
        endTime: TimeInterval
    ) async throws -> URL
}

@MainActor
protocol FlashbackGenerationServicing {
    func generateHighlights(for session: RecordingSession, context: ModelContext) async throws -> [HighlightClip]
}

extension ClipExtractionService: ClipExtractionServicing {}

@MainActor
final class FlashbackGenerationService: FlashbackGenerationServicing {
    private let clipExtractor: ClipExtractionServicing
    private let calendar = Calendar.current

    init(clipExtractor: ClipExtractionServicing = ClipExtractionService()) {
        self.clipExtractor = clipExtractor
    }

    func generateHighlights(for session: RecordingSession, context: ModelContext) async throws -> [HighlightClip] {
        guard session.duration > 0 else { return [] }

        let windowConfiguration = HighlightWindowBuilder.Configuration(
            preRoll: 3,
            postRoll: 5,
            burstGap: 1.1,
            mergeGap: 0.9,
            minClipDuration: 6,
            minConfidence: 0.45,
            maxConfidencePostBoost: 1.25
        )

        let windows = HighlightWindowBuilder.buildWindows(
            events: session.laughterEvents.map {
                HighlightWindowBuilder.DetectionPoint(
                    offset: $0.timestampOffset,
                    confidence: $0.confidence
                )
            },
            sessionDuration: session.duration,
            configuration: windowConfiguration
        )

        guard !windows.isEmpty else { return [] }

        var clips: [HighlightClip] = []
        let sourceURL = session.audioFileURL
        let dayBucket = calendar.startOfDay(for: session.startedAt)

        for (index, window) in windows.enumerated() {
            let destination = try AppFileStore.newClipURL(sessionID: session.id, index: index)
            let clipURL = try await clipExtractor.extractClip(
                from: sourceURL,
                to: destination,
                startTime: window.lowerBound,
                endTime: window.upperBound
            )

            let clip = HighlightClip(
                sourceStartOffset: window.lowerBound,
                sourceEndOffset: window.upperBound,
                clipFilePath: clipURL.path,
                dayBucket: dayBucket,
                session: session
            )

            session.highlightClips.append(clip)
            context.insert(clip)
            clips.append(clip)
        }

        try context.save()
        return clips
    }
}
