import AuthenticationServices
import SwiftUI

struct AuthGateView: View {
    @EnvironmentObject private var authSession: AuthSession

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.07, green: 0.10, blue: 0.20), Color(red: 0.12, green: 0.44, blue: 0.67)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 24) {
                Spacer()

                Text("Memento")
                    .font(.system(size: 52, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text("Capture conversations, detect laughter, and relive the best moments tomorrow.")
                    .font(.title3.weight(.medium))
                    .foregroundStyle(.white.opacity(0.92))

                SignInWithAppleButton(.continue, onRequest: { request in
                    request.requestedScopes = [.fullName, .email]
                }, onCompletion: { result in
                    authSession.handleSignIn(result: result)
                })
                .signInWithAppleButtonStyle(.white)
                .frame(height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                Spacer()
            }
            .padding(24)
        }
    }
}
