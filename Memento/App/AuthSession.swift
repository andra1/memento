import AuthenticationServices
import Foundation

@MainActor
final class AuthSession: NSObject, ObservableObject {
    @Published var isAuthenticated: Bool
    @Published var displayName: String

    private let authKey = "memento.authenticated"
    private let nameKey = "memento.displayName"

    override init() {
        self.isAuthenticated = UserDefaults.standard.bool(forKey: authKey)
        self.displayName = UserDefaults.standard.string(forKey: nameKey) ?? ""
        super.init()
    }

    func handleSignIn(result: Result<ASAuthorization, Error>) {
        switch result {
        case let .success(authResults):
            guard let credential = authResults.credential as? ASAuthorizationAppleIDCredential else { return }
            isAuthenticated = true
            UserDefaults.standard.set(true, forKey: authKey)

            let formatter = PersonNameComponentsFormatter()
            if let fullName = credential.fullName {
                let resolvedName = formatter.string(from: fullName)
                if !resolvedName.isEmpty {
                    displayName = resolvedName
                    UserDefaults.standard.set(resolvedName, forKey: nameKey)
                }
            }
        case .failure:
            break
        }
    }

    func signOut() {
        isAuthenticated = false
        displayName = ""
        UserDefaults.standard.removeObject(forKey: authKey)
        UserDefaults.standard.removeObject(forKey: nameKey)
    }
}
