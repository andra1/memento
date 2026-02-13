import SwiftUI

struct RecorderView: View {
    @Environment(\.modelContext) private var modelContext
    @StateObject private var viewModel = RecorderViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                timerCard
                laughterMarkersCard
                recordButton
            }
            .padding()
        }
        .navigationTitle("Record")
        .alert("Recorder", isPresented: errorAlertBinding, actions: {
            Button("OK") {
                viewModel.errorMessage = nil
            }
        }, message: {
            Text(viewModel.errorMessage ?? "")
        })
    }

    private var timerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(viewModel.isRecording ? "Recording Live" : "Ready to Record")
                .font(.headline)

            Text(formattedTime(viewModel.elapsedTime))
                .font(.system(size: 42, weight: .bold, design: .rounded))
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private var laughterMarkersCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Laughter Markers")
                .font(.headline)

            if viewModel.markers.isEmpty {
                Text("Detected laughter moments will appear here.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.markers.suffix(12)) { marker in
                    HStack {
                        Image(systemName: "face.smiling")
                            .foregroundStyle(.orange)
                        Text("\(marker.offset.formatted(.number.precision(.fractionLength(1))))s")
                        Spacer()
                        Text("\(Int(marker.confidence * 100))%")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private var recordButton: some View {
        Button {
            Task {
                if viewModel.isRecording {
                    await viewModel.stopRecording()
                } else {
                    await viewModel.startRecording(context: modelContext)
                }
            }
        } label: {
            Text(viewModel.isRecording ? "Stop" : "Start Recording")
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isRecording ? Color.red : Color.blue)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
    }

    private func formattedTime(_ interval: TimeInterval) -> String {
        let total = Int(interval)
        let minutes = total / 60
        let seconds = total % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    private var errorAlertBinding: Binding<Bool> {
        Binding(
            get: { viewModel.errorMessage != nil },
            set: { isPresented in
                if !isPresented {
                    viewModel.errorMessage = nil
                }
            }
        )
    }
}
