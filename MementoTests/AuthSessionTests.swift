import XCTest
@testable import Memento

@MainActor
final class AuthSessionTests: XCTestCase {
    private let authKey = "memento.authenticated"
    private let nameKey = "memento.displayName"

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removeObject(forKey: authKey)
        UserDefaults.standard.removeObject(forKey: nameKey)
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: authKey)
        UserDefaults.standard.removeObject(forKey: nameKey)
        super.tearDown()
    }

    func testInitializesFromUserDefaults() {
        UserDefaults.standard.set(true, forKey: authKey)
        UserDefaults.standard.set("Kaushik", forKey: nameKey)

        let session = AuthSession()

        XCTAssertTrue(session.isAuthenticated)
        XCTAssertEqual(session.displayName, "Kaushik")
    }

    func testSignOutClearsStateAndPersistence() {
        UserDefaults.standard.set(true, forKey: authKey)
        UserDefaults.standard.set("Kaushik", forKey: nameKey)
        let session = AuthSession()

        session.signOut()

        XCTAssertFalse(session.isAuthenticated)
        XCTAssertEqual(session.displayName, "")
        XCTAssertFalse(UserDefaults.standard.bool(forKey: authKey))
        XCTAssertNil(UserDefaults.standard.string(forKey: nameKey))
    }
}
