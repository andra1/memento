import SwiftData
@testable import Memento

struct TestModelStack {
    let container: ModelContainer
    let context: ModelContext

    static func make() throws -> TestModelStack {
        let schema = Schema([
            RecordingSession.self,
            LaughterEvent.self,
            HighlightClip.self,
            UserSettings.self
        ])
        let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: schema, configurations: [configuration])
        return TestModelStack(container: container, context: ModelContext(container))
    }
}
