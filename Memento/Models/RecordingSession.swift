import Foundation
import SwiftData

@Model
final class RecordingSession {
    @Attribute(.unique) var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var audioFilePath: String
    var duration: TimeInterval

    @Relationship(deleteRule: .cascade, inverse: \LaughterEvent.session)
    var laughterEvents: [LaughterEvent]

    @Relationship(deleteRule: .cascade, inverse: \HighlightClip.session)
    var highlightClips: [HighlightClip]

    init(
        id: UUID = UUID(),
        startedAt: Date = Date(),
        endedAt: Date? = nil,
        audioFilePath: String,
        duration: TimeInterval = 0,
        laughterEvents: [LaughterEvent] = [],
        highlightClips: [HighlightClip] = []
    ) {
        self.id = id
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.audioFilePath = audioFilePath
        self.duration = duration
        self.laughterEvents = laughterEvents
        self.highlightClips = highlightClips
    }

    var audioFileURL: URL {
        URL(fileURLWithPath: audioFilePath)
    }
}
