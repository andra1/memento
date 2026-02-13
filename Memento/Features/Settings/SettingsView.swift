import SwiftData
import SwiftUI

struct SettingsView: View {
    @Environment(\.modelContext) private var modelContext
    @EnvironmentObject private var notificationCoordinator: NotificationCoordinator

    @Query private var settingsList: [UserSettings]

    private var settings: UserSettings {
        if let first = settingsList.first {
            return first
        }
        let created = UserSettings()
        modelContext.insert(created)
        try? modelContext.save()
        return created
    }

    var body: some View {
        Form {
            Section("Account") {
                Text("Authentication is currently disabled.")
                Text("Running in local mode for core app testing.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Flashback") {
                Toggle("Daily Notification", isOn: bindingForNotifications)
                Stepper(value: bindingForRetention, in: 7 ... 365, step: 1) {
                    Text("Retention: \(settings.retentionDays) days")
                }
            }

            Section("Privacy") {
                Text("Audio stays on-device for MVP. You can delete any clip from Highlights.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
    }

    private var bindingForNotifications: Binding<Bool> {
        Binding(
            get: { settings.notificationsEnabled },
            set: { newValue in
                settings.notificationsEnabled = newValue
                try? modelContext.save()
                Task {
                    await notificationCoordinator.refreshDailyNotification(enabled: newValue)
                }
            }
        )
    }

    private var bindingForRetention: Binding<Int> {
        Binding(
            get: { settings.retentionDays },
            set: { newValue in
                settings.retentionDays = newValue
                try? modelContext.save()
            }
        )
    }
}
