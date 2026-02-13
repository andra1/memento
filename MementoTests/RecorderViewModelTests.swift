import AVFoundation
import Foundation
import SwiftData
import XCTest
@testable import Memento

@MainActor
final class RecorderViewModelTests: XCTestCase {
    private enum TestStartError: LocalizedError {
        case failed

        var errorDescription: String? {
            "start failed"
        }
    }

    private final class MockRecordingService: AudioRecordingServicing {
        var currentDuration: TimeInterval = 0
        var permissionGranted = true
        var startError: Error?
        private(set) var didStart = false
        private(set) var didStop = false
        private(set) var outputURL: URL?

        func requestRecordPermission() async -> Bool {
            permissionGranted
        }

        func startRecording(
            to outputURL: URL,
            onBuffer: @escaping (AVAudioPCMBuffer, AVAudioTime) -> Void
        ) throws {
            didStart = true
            self.outputURL = outputURL
            if let startError {
                throw startError
            }
        }

        func stopRecording() {
            didStop = true
        }
    }

    private final class MockLaughterService: LaughterDetectionServicing {
        var onLaughterDetected: ((TimeInterval, Double) -> Void)?

        func analyze(buffer: AVAudioPCMBuffer, when: AVAudioTime) {
            // No-op for unit tests.
        }

        func emit(offset: TimeInterval, confidence: Double) {
            onLaughterDetected?(offset, confidence)
        }
    }

    private final class MockFlashbackService: FlashbackGenerationServicing {
        var generatedSessionIDs: [UUID] = []

        func generateHighlights(for session: RecordingSession, context: ModelContext) async throws -> [HighlightClip] {
            generatedSessionIDs.append(session.id)
            return []
        }
    }

    func testStartRecordingPermissionDeniedSetsError() async throws {
        let stack = try TestModelStack.make()
        let recording = MockRecordingService()
        recording.permissionGranted = false
        let laughter = MockLaughterService()
        let flashback = MockFlashbackService()
        let viewModel = RecorderViewModel(
            recordingService: recording,
            laughterService: laughter,
            flashbackService: flashback,
            makeRecordingURL: { self.makeTempRecordingURL() }
        )

        await viewModel.startRecording(context: stack.context)

        XCTAssertFalse(viewModel.isRecording)
        XCTAssertEqual(viewModel.errorMessage, "Microphone access is denied. Enable it in Settings > Privacy & Security > Microphone for Memento.")
        XCTAssertTrue(viewModel.shouldShowSettingsShortcut)
        XCTAssertFalse(recording.didStart)
        XCTAssertEqual(try stack.context.fetch(FetchDescriptor<RecordingSession>()).count, 0)
    }

    func testStartRecordingSuccessCreatesSession() async throws {
        let stack = try TestModelStack.make()
        let recording = MockRecordingService()
        let laughter = MockLaughterService()
        let flashback = MockFlashbackService()
        let outputURL = makeTempRecordingURL()
        let viewModel = RecorderViewModel(
            recordingService: recording,
            laughterService: laughter,
            flashbackService: flashback,
            makeRecordingURL: { outputURL }
        )

        await viewModel.startRecording(context: stack.context)

        XCTAssertTrue(viewModel.isRecording)
        XCTAssertNil(viewModel.errorMessage)
        XCTAssertTrue(recording.didStart)
        XCTAssertEqual(recording.outputURL, outputURL)

        let sessions = try stack.context.fetch(FetchDescriptor<RecordingSession>())
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].audioFilePath, outputURL.path)

        await viewModel.stopRecording()
    }

    func testStartRecordingFailureShowsUnderlyingError() async throws {
        let stack = try TestModelStack.make()
        let recording = MockRecordingService()
        recording.startError = TestStartError.failed
        let viewModel = RecorderViewModel(
            recordingService: recording,
            laughterService: MockLaughterService(),
            flashbackService: MockFlashbackService(),
            makeRecordingURL: { self.makeTempRecordingURL() }
        )

        await viewModel.startRecording(context: stack.context)

        XCTAssertFalse(viewModel.isRecording)
        XCTAssertTrue(viewModel.errorMessage?.contains("start failed") == true)
        XCTAssertTrue(recording.didStop)
        XCTAssertEqual(try stack.context.fetch(FetchDescriptor<RecordingSession>()).count, 0)
    }

    func testStopRecordingPersistsSessionEndAndDuration() async throws {
        let stack = try TestModelStack.make()
        let recording = MockRecordingService()
        recording.currentDuration = 3.5
        let flashback = MockFlashbackService()
        let viewModel = RecorderViewModel(
            recordingService: recording,
            laughterService: MockLaughterService(),
            flashbackService: flashback,
            makeRecordingURL: { self.makeTempRecordingURL() }
        )

        await viewModel.startRecording(context: stack.context)
        await viewModel.stopRecording()

        XCTAssertFalse(viewModel.isRecording)
        XCTAssertTrue(recording.didStop)

        let sessions = try stack.context.fetch(FetchDescriptor<RecordingSession>())
        XCTAssertEqual(sessions.count, 1)
        XCTAssertNotNil(sessions[0].endedAt)
        XCTAssertEqual(sessions[0].duration, 3.5, accuracy: 0.001)
        XCTAssertEqual(flashback.generatedSessionIDs, [sessions[0].id])
    }

    func testLaughterCallbackAddsMarkerAndPersistsEvent() async throws {
        let stack = try TestModelStack.make()
        let recording = MockRecordingService()
        let laughter = MockLaughterService()
        let viewModel = RecorderViewModel(
            recordingService: recording,
            laughterService: laughter,
            flashbackService: MockFlashbackService(),
            makeRecordingURL: { self.makeTempRecordingURL() }
        )

        await viewModel.startRecording(context: stack.context)
        laughter.emit(offset: 1.2, confidence: 0.88)
        await Task.yield()
        await Task.yield()
        await viewModel.stopRecording()

        XCTAssertEqual(viewModel.markers.count, 1)
        XCTAssertEqual(viewModel.markers[0].offset, 1.2, accuracy: 0.001)
        XCTAssertEqual(viewModel.markers[0].confidence, 0.88, accuracy: 0.001)

        let events = try stack.context.fetch(FetchDescriptor<LaughterEvent>())
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].timestampOffset, 1.2, accuracy: 0.001)
        XCTAssertEqual(events[0].confidence, 0.88, accuracy: 0.001)
    }

    private func makeTempRecordingURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("recording-\(UUID().uuidString).caf")
    }
}
