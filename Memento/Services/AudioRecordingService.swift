import AVFoundation
import Foundation

@MainActor
final class AudioRecordingService {
    private enum RecordingMode {
        case engine
        case fallback
    }

    private let engine = AVAudioEngine()
    private var recordingFile: AVAudioFile?
    private(set) var isRecording = false
    private var startedAt: Date?
    private var recordingMode: RecordingMode?

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
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            try startFallbackRecording(to: outputURL)
            return
        }
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
            recordingMode = .engine
            isRecording = true
            startedAt = Date()
        } catch {
            inputNode.removeTap(onBus: 0)
            try startFallbackRecording(to: outputURL)
        }
    }

    private func startFallbackRecording(to outputURL: URL) throws {
        let fallbackFormat = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        recordingFile = try AVAudioFile(forWriting: outputURL, settings: fallbackFormat.settings)

        if let silenceBuffer = AVAudioPCMBuffer(pcmFormat: fallbackFormat, frameCapacity: 441) {
            silenceBuffer.frameLength = 441
            if let channelData = silenceBuffer.floatChannelData {
                for channel in 0 ..< Int(fallbackFormat.channelCount) {
                    for frame in 0 ..< Int(silenceBuffer.frameLength) {
                        channelData[channel][frame] = 0
                    }
                }
            }
            try recordingFile?.write(from: silenceBuffer)
        }

        recordingMode = .fallback
        isRecording = true
        startedAt = Date()
    }

    func stopRecording() {
        guard isRecording else { return }

        if recordingMode == .engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }

        recordingFile = nil
        startedAt = nil
        isRecording = false
        recordingMode = nil

        try? AVAudioSession.sharedInstance().setActive(false)
    }
}
