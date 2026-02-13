import Foundation
import SwiftData
import XCTest
@testable import Memento

@MainActor
final class FlashbackGenerationServiceTests: XCTestCase {
    private final class MockClipExtractor: ClipExtractionServicing {
        struct Call {
            let sourceURL: URL
            let outputURL: URL
            let startTime: TimeInterval
            let endTime: TimeInterval
        }

        var calls: [Call] = []

        func extractClip(
            from sourceURL: URL,
            to outputURL: URL,
            startTime: TimeInterval,
            endTime: TimeInterval
        ) async throws -> URL {
            calls.append(Call(sourceURL: sourceURL, outputURL: outputURL, startTime: startTime, endTime: endTime))
            try Data("clip".utf8).write(to: outputURL, options: .atomic)
            return outputURL
        }
    }

    func testGenerateHighlightsReturnsEmptyWhenDurationIsZero() async throws {
        let stack = try TestModelStack.make()
        let extractor = MockClipExtractor()
        let service = FlashbackGenerationService(clipExtractor: extractor)
        let session = RecordingSession(audioFilePath: "/tmp/empty.caf", duration: 0)
        stack.context.insert(session)
        try stack.context.save()

        let clips = try await service.generateHighlights(for: session, context: stack.context)

        XCTAssertTrue(clips.isEmpty)
        XCTAssertTrue(extractor.calls.isEmpty)
    }

    func testGenerateHighlightsSkipsEventsBelowConfidenceThreshold() async throws {
        let stack = try TestModelStack.make()
        let extractor = MockClipExtractor()
        let service = FlashbackGenerationService(clipExtractor: extractor)
        let session = RecordingSession(audioFilePath: "/tmp/source.caf", duration: 30)
        stack.context.insert(session)
        try stack.context.save()

        let lowConfidence = LaughterEvent(timestampOffset: 10, confidence: 0.2, session: session)
        session.laughterEvents.append(lowConfidence)
        stack.context.insert(lowConfidence)
        try stack.context.save()

        let clips = try await service.generateHighlights(for: session, context: stack.context)

        XCTAssertTrue(clips.isEmpty)
        XCTAssertTrue(extractor.calls.isEmpty)
    }

    func testGenerateHighlightsCreatesClipAndPersistsIt() async throws {
        let stack = try TestModelStack.make()
        let extractor = MockClipExtractor()
        let service = FlashbackGenerationService(clipExtractor: extractor)

        let session = RecordingSession(audioFilePath: "/tmp/source.caf", duration: 30)
        stack.context.insert(session)
        try stack.context.save()

        let laugh = LaughterEvent(timestampOffset: 8, confidence: 0.95, session: session)
        session.laughterEvents.append(laugh)
        stack.context.insert(laugh)
        try stack.context.save()

        let clips = try await service.generateHighlights(for: session, context: stack.context)

        XCTAssertEqual(clips.count, 1)
        XCTAssertEqual(extractor.calls.count, 1)
        XCTAssertEqual(session.highlightClips.count, 1)
        XCTAssertLessThanOrEqual(clips[0].sourceStartOffset, 8)
        XCTAssertGreaterThan(clips[0].sourceEndOffset, 8)

        let persisted = try stack.context.fetch(FetchDescriptor<HighlightClip>())
        XCTAssertEqual(persisted.count, 1)
    }
}
