import SwiftUI

struct RootView: View {
    var body: some View {
        MainTabView()
    }
}

struct MainTabView: View {
    @Environment(\.modelContext) private var modelContext
    @StateObject private var orchestrator = FlashbackOrchestrator()

    var body: some View {
        TabView {
            NavigationStack {
                HomeView()
            }
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }

            NavigationStack {
                RecorderView()
            }
            .tabItem {
                Label("Record", systemImage: "mic.fill")
            }

            NavigationStack {
                HighlightsListView()
            }
            .tabItem {
                Label("Highlights", systemImage: "sparkles")
            }

            NavigationStack {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape.fill")
            }
        }
        .task {
            await orchestrator.generateYesterdayIfNeeded(context: modelContext)
        }
    }
}
