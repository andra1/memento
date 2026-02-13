import Foundation

enum AppFileStore {
    private static let recordingsFolder = "Recordings"
    private static let clipsFolder = "Clips"

    static func recordingsDirectory() throws -> URL {
        try ensureDirectory(named: recordingsFolder)
    }

    static func clipsDirectory() throws -> URL {
        try ensureDirectory(named: clipsFolder)
    }

    static func newRecordingURL() throws -> URL {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        // Raw engine taps write PCM buffers, so use a PCM-friendly container.
        let filename = "recording-\(formatter.string(from: Date())).caf"
        return try recordingsDirectory().appendingPathComponent(filename)
    }

    static func newClipURL(sessionID: UUID, index: Int) throws -> URL {
        let filename = "clip-\(sessionID.uuidString)-\(index).m4a"
        return try clipsDirectory().appendingPathComponent(filename)
    }

    private static func ensureDirectory(named folder: String) throws -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = base.appendingPathComponent(folder, isDirectory: true)
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }
}
