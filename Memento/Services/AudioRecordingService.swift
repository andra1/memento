import AVFoundation
import Foundation

@MainActor
final class AudioRecordingService {
    enum RecordingError: LocalizedError {
        case permissionDenied
        case microphoneUnavailable
        case failedToStart(Error)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone access is denied. Enable it in Settings and try again."
            case .microphoneUnavailable:
                return "Microphone input is unavailable on this device."
            case let .failedToStart(error):
                return "Unable to start microphone capture: \(error.localizedDescription)"
            }
        }
    }

    private let engine = AVAudioEngine()
    private var recordingFile: AVAudioFile?
    private(set) var isRecording = false
    private var startedAt: Date?

    var currentDuration: TimeInterval {
        guard let startedAt else { return 0 }
        return max(0, Date().timeIntervalSince(startedAt))
    }

    func requestRecordPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            if #available(iOS 17, *) {
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        }
    }

    func startRecording(
        to outputURL: URL,
        onBuffer: @escaping (AVAudioPCMBuffer, AVAudioTime) -> Void
    ) throws {
        let session = AVAudioSession.sharedInstance()
        guard session.recordPermission == .granted else {
            throw RecordingError.permissionDenied
        }

        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true)
        guard session.isInputAvailable else {
            throw RecordingError.microphoneUnavailable
        }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw RecordingError.microphoneUnavailable
        }

        engine.stop()
        engine.reset()
        recordingFile = try AVAudioFile(forWriting: outputURL, settings: inputFormat.settings)

        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 2_048, format: inputFormat) { [weak self] buffer, when in
            do {
                try self?.recordingFile?.write(from: buffer)
            } catch {
                return
            }
            onBuffer(buffer, when)
        }

        do {
            engine.prepare()
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            recordingFile = nil
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw RecordingError.failedToStart(error)
        }
        isRecording = true
        startedAt = Date()
    }

    func stopRecording() {
        guard isRecording else { return }

        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning {
            engine.stop()
        }
        engine.reset()

        recordingFile = nil
        startedAt = nil
        isRecording = false

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
