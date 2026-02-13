import Foundation
import SwiftData
import XCTest
@testable import Memento

@MainActor
final class FlashbackOrchestratorTests: XCTestCase {
    private final class MockGenerator: FlashbackGenerationServicing {
        var generatedSessionIDs: [UUID] = []

        func generateHighlights(for session: RecordingSession, context: ModelContext) async throws -> [HighlightClip] {
            generatedSessionIDs.append(session.id)
            return []
        }
    }

    private let defaultsKey = "test.lastFlashbackGenerationDate"
    private var defaultsSuiteName: String!
    private var defaults: UserDefaults!
    private var calendar: Calendar!

    override func setUp() {
        super.setUp()
        defaultsSuiteName = "FlashbackOrchestratorTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: defaultsSuiteName)
        defaults.removePersistentDomain(forName: defaultsSuiteName)

        var configured = Calendar(identifier: .gregorian)
        configured.timeZone = TimeZone(secondsFromGMT: 0)!
        calendar = configured
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: defaultsSuiteName)
        defaults = nil
        defaultsSuiteName = nil
        calendar = nil
        super.tearDown()
    }

    func testSkipsWhenAlreadyGeneratedToday() async throws {
        let stack = try TestModelStack.make()
        let generator = MockGenerator()
        let now = makeDate(year: 2026, month: 2, day: 13, hour: 12)
        defaults.set(now, forKey: defaultsKey)

        let orchestrator = FlashbackOrchestrator(
            generator: generator,
            userDefaults: defaults,
            defaultsKey: defaultsKey,
            calendar: calendar,
            now: { now }
        )

        await orchestrator.generateYesterdayIfNeeded(context: stack.context)

        XCTAssertTrue(generator.generatedSessionIDs.isEmpty)
    }

    func testGeneratesOnlyForYesterdaySessionsWithoutHighlights() async throws {
        let stack = try TestModelStack.make()
        let generator = MockGenerator()
        let now = makeDate(year: 2026, month: 2, day: 13, hour: 12)
        let yesterday = makeDate(year: 2026, month: 2, day: 12, hour: 9)
        let today = makeDate(year: 2026, month: 2, day: 13, hour: 8)

        let target = RecordingSession(startedAt: yesterday, audioFilePath: "/tmp/a.caf", duration: 12)
        let laugh = LaughterEvent(timestampOffset: 2, confidence: 0.8, session: target)
        target.laughterEvents.append(laugh)

        let noLaughter = RecordingSession(startedAt: yesterday, audioFilePath: "/tmp/b.caf", duration: 12)
        let alreadyHighlighted = RecordingSession(startedAt: yesterday, audioFilePath: "/tmp/c.caf", duration: 12)
        let clip = HighlightClip(
            sourceStartOffset: 0,
            sourceEndOffset: 4,
            clipFilePath: "/tmp/clip.m4a",
            dayBucket: calendar.startOfDay(for: yesterday),
            session: alreadyHighlighted
        )
        alreadyHighlighted.highlightClips.append(clip)

        let notYesterday = RecordingSession(startedAt: today, audioFilePath: "/tmp/d.caf", duration: 12)
        let todayLaugh = LaughterEvent(timestampOffset: 1, confidence: 0.7, session: notYesterday)
        notYesterday.laughterEvents.append(todayLaugh)

        stack.context.insert(target)
        stack.context.insert(laugh)
        stack.context.insert(noLaughter)
        stack.context.insert(alreadyHighlighted)
        stack.context.insert(clip)
        stack.context.insert(notYesterday)
        stack.context.insert(todayLaugh)
        try stack.context.save()

        let orchestrator = FlashbackOrchestrator(
            generator: generator,
            userDefaults: defaults,
            defaultsKey: defaultsKey,
            calendar: calendar,
            now: { now }
        )

        await orchestrator.generateYesterdayIfNeeded(context: stack.context)

        XCTAssertEqual(generator.generatedSessionIDs, [target.id])
        XCTAssertTrue(calendar.isDate(defaults.object(forKey: defaultsKey) as? Date ?? .distantPast, inSameDayAs: now))
    }

    private func makeDate(year: Int, month: Int, day: Int, hour: Int) -> Date {
        var components = DateComponents()
        components.year = year
        components.month = month
        components.day = day
        components.hour = hour
        components.minute = 0
        components.second = 0
        return calendar.date(from: components)!
    }
}
