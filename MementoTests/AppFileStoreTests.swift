import XCTest
@testable import Memento

final class AppFileStoreTests: XCTestCase {
    func testRecordingsDirectoryExists() throws {
        let directory = try AppFileStore.recordingsDirectory()
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)

        XCTAssertTrue(exists)
        XCTAssertTrue(isDirectory.boolValue)
    }

    func testNewRecordingURLUsesCAFExtension() throws {
        let url = try AppFileStore.newRecordingURL()

        XCTAssertEqual(url.pathExtension, "caf")
        XCTAssertTrue(url.lastPathComponent.hasPrefix("recording-"))
    }

    func testNewClipURLUsesM4AExtension() throws {
        let sessionID = UUID()
        let url = try AppFileStore.newClipURL(sessionID: sessionID, index: 3)

        XCTAssertEqual(url.pathExtension, "m4a")
        XCTAssertTrue(url.lastPathComponent.contains(sessionID.uuidString))
        XCTAssertTrue(url.lastPathComponent.hasSuffix("-3.m4a"))
    }
}
