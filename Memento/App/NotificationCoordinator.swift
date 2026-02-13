import Foundation
import UserNotifications

@MainActor
final class NotificationCoordinator: ObservableObject {
    private let center = UNUserNotificationCenter.current()
    private let dailyIdentifier = "memento.flashback.daily"

    func refreshDailyNotification(enabled: Bool) async {
        if enabled {
            do {
                let granted = try await center.requestAuthorization(options: [.alert, .badge, .sound])
                if granted {
                    scheduleDailyFlashbackNotification()
                }
            } catch {
                return
            }
        } else {
            center.removePendingNotificationRequests(withIdentifiers: [dailyIdentifier])
        }
    }

    private func scheduleDailyFlashbackNotification() {
        center.removePendingNotificationRequests(withIdentifiers: [dailyIdentifier])

        let content = UNMutableNotificationContent()
        content.title = "Your Memento flashback is ready"
        content.body = "Revisit yesterday's laughter highlights."
        content.sound = .default

        var components = DateComponents()
        components.hour = 9
        components.minute = 0

        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
        let request = UNNotificationRequest(identifier: dailyIdentifier, content: content, trigger: trigger)
        center.add(request)
    }
}
