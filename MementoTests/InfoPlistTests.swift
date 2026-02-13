import XCTest
@testable import Memento

final class InfoPlistTests: XCTestCase {
    func testMicrophoneUsageDescriptionExistsAndIsNonEmpty() {
        let appBundle = Bundle(for: RecorderViewModel.self)
        let value = appBundle.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") as? String

        XCTAssertNotNil(value)
        XCTAssertFalse(value?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
    }
}
