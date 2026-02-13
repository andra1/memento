import SwiftData
import SwiftUI

@main
struct MementoApp: App {
    @StateObject private var authSession = AuthSession()
    @StateObject private var notificationCoordinator = NotificationCoordinator()

    private let modelContainer: ModelContainer = {
        do {
            let schema = Schema([
                RecordingSession.self,
                LaughterEvent.self,
                HighlightClip.self,
                UserSettings.self
            ])
            let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
            return try ModelContainer(for: schema, configurations: [configuration])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(authSession)
                .environmentObject(notificationCoordinator)
        }
        .modelContainer(modelContainer)
    }
}
