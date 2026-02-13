import XCTest
@testable import Memento

final class ModelURLTests: XCTestCase {
    func testRecordingSessionAudioFileURLUsesStoredPath() {
        let session = RecordingSession(audioFilePath: "/tmp/session-audio.caf")

        XCTAssertEqual(session.audioFileURL.path, "/tmp/session-audio.caf")
    }

    func testHighlightClipFileURLUsesStoredPath() {
        let clip = HighlightClip(
            sourceStartOffset: 0,
            sourceEndOffset: 3,
            clipFilePath: "/tmp/highlight.m4a",
            dayBucket: Date()
        )

        XCTAssertEqual(clip.clipFileURL.path, "/tmp/highlight.m4a")
    }
}
