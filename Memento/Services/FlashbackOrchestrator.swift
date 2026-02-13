import Foundation
import SwiftData

@MainActor
final class FlashbackOrchestrator: ObservableObject {
    private let generator: FlashbackGenerationServicing
    private let userDefaults: UserDefaults
    private let defaultsKey: String
    private let calendar: Calendar
    private let now: () -> Date

    init(
        generator: FlashbackGenerationServicing = FlashbackGenerationService(),
        userDefaults: UserDefaults = .standard,
        defaultsKey: String = "memento.lastFlashbackGenerationDate",
        calendar: Calendar = .current,
        now: @escaping () -> Date = Date.init
    ) {
        self.generator = generator
        self.userDefaults = userDefaults
        self.defaultsKey = defaultsKey
        self.calendar = calendar
        self.now = now
    }

    func generateYesterdayIfNeeded(context: ModelContext) async {
        let today = calendar.startOfDay(for: now())
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today) ?? today

        if let lastRun = userDefaults.object(forKey: defaultsKey) as? Date,
           calendar.isDate(lastRun, inSameDayAs: today) {
            return
        }

        do {
            let sessions = try context.fetch(FetchDescriptor<RecordingSession>())
            let targets = sessions.filter { session in
                session.startedAt >= yesterday && session.startedAt < today &&
                !session.laughterEvents.isEmpty &&
                session.highlightClips.isEmpty
            }

            for session in targets {
                _ = try await generator.generateHighlights(for: session, context: context)
            }

            userDefaults.set(now(), forKey: defaultsKey)
        } catch {
            return
        }
    }
}
