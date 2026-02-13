import Foundation

struct LaughterMarker: Identifiable, Hashable {
    let id = UUID()
    let offset: TimeInterval
    let confidence: Double
}
