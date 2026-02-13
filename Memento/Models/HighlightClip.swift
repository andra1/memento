import Foundation
import SwiftData

@Model
final class HighlightClip {
    @Attribute(.unique) var id: UUID
    var sourceStartOffset: TimeInterval
    var sourceEndOffset: TimeInterval
    var createdAt: Date
    var clipFilePath: String
    var isFavorited: Bool
    var dayBucket: Date

    var session: RecordingSession?

    init(
        id: UUID = UUID(),
        sourceStartOffset: TimeInterval,
        sourceEndOffset: TimeInterval,
        createdAt: Date = Date(),
        clipFilePath: String,
        isFavorited: Bool = false,
        dayBucket: Date,
        session: RecordingSession? = nil
    ) {
        self.id = id
        self.sourceStartOffset = sourceStartOffset
        self.sourceEndOffset = sourceEndOffset
        self.createdAt = createdAt
        self.clipFilePath = clipFilePath
        self.isFavorited = isFavorited
        self.dayBucket = dayBucket
        self.session = session
    }

    var clipFileURL: URL {
        URL(fileURLWithPath: clipFilePath)
    }
}
