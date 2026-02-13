import Foundation
import SwiftData

@Model
final class UserSettings {
    @Attribute(.unique) var id: UUID
    var notificationsEnabled: Bool
    var retentionDays: Int

    init(
        id: UUID = UUID(),
        notificationsEnabled: Bool = true,
        retentionDays: Int = 30
    ) {
        self.id = id
        self.notificationsEnabled = notificationsEnabled
        self.retentionDays = retentionDays
    }
}
