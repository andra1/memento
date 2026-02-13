import Foundation
import SwiftData

@Model
final class LaughterEvent {
    @Attribute(.unique) var id: UUID
    var timestampOffset: TimeInterval
    var confidence: Double
    var detectedAt: Date

    var session: RecordingSession?

    init(
        id: UUID = UUID(),
        timestampOffset: TimeInterval,
        confidence: Double,
        detectedAt: Date = Date(),
        session: RecordingSession? = nil
    ) {
        self.id = id
        self.timestampOffset = timestampOffset
        self.confidence = confidence
        self.detectedAt = detectedAt
        self.session = session
    }
}
