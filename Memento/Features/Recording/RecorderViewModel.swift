import AVFoundation
import Foundation
import SwiftData

@MainActor
protocol AudioRecordingServicing: AnyObject {
    var currentDuration: TimeInterval { get }
    func requestRecordPermission() async -> Bool
    func startRecording(
        to outputURL: URL,
        onBuffer: @escaping (AVAudioPCMBuffer, AVAudioTime) -> Void
    ) throws
    func stopRecording()
}

protocol LaughterDetectionServicing: AnyObject {
    var onLaughterDetected: ((TimeInterval, Double) -> Void)? { get set }
    func analyze(buffer: AVAudioPCMBuffer, when: AVAudioTime)
}

extension AudioRecordingService: AudioRecordingServicing {}
extension LaughterDetectionService: LaughterDetectionServicing {}

@MainActor
final class RecorderViewModel: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var elapsedTime: TimeInterval = 0
    @Published private(set) var markers: [LaughterMarker] = []
    @Published var shouldShowSettingsShortcut = false
    @Published var errorMessage: String?

    private let recordingService: AudioRecordingServicing
    private let laughterService: LaughterDetectionServicing
    private let flashbackService: FlashbackGenerationServicing
    private let isRecordPermissionDenied: () -> Bool
    private let makeRecordingURL: () throws -> URL

    private var activeSession: RecordingSession?
    private var timer: Timer?
    private var modelContext: ModelContext?

    init(
        recordingService: AudioRecordingServicing = AudioRecordingService(),
        laughterService: LaughterDetectionServicing = LaughterDetectionService(),
        flashbackService: FlashbackGenerationServicing = FlashbackGenerationService(),
        isRecordPermissionDenied: @escaping () -> Bool = {
            AVAudioApplication.shared.recordPermission == .denied
        },
        makeRecordingURL: @escaping () throws -> URL = { try AppFileStore.newRecordingURL() }
    ) {
        self.recordingService = recordingService
        self.laughterService = laughterService
        self.flashbackService = flashbackService
        self.isRecordPermissionDenied = isRecordPermissionDenied
        self.makeRecordingURL = makeRecordingURL

        laughterService.onLaughterDetected = { [weak self] offset, confidence in
            guard let self else { return }
            Task { @MainActor in
                await self.handleLaughter(offset: offset, confidence: confidence)
            }
        }
    }

    func startRecording(context: ModelContext) async {
        guard !isRecording else { return }
        modelContext = context
        errorMessage = nil
        shouldShowSettingsShortcut = false

        let granted = await recordingService.requestRecordPermission()
        guard granted else {
            if isRecordPermissionDenied() {
                errorMessage = "Microphone access is denied. Enable it in Settings > Privacy & Security > Microphone for Memento."
                shouldShowSettingsShortcut = true
            } else {
                errorMessage = "Microphone permission is required to record conversations."
            }
            return
        }

        do {
            let outputURL = try makeRecordingURL()
            let session = RecordingSession(audioFilePath: outputURL.path)

            try recordingService.startRecording(to: outputURL) { [weak self] buffer, when in
                self?.laughterService.analyze(buffer: buffer, when: when)
            }

            context.insert(session)
            try context.save()

            activeSession = session
            isRecording = true
            elapsedTime = 0
            markers = []
            startTimer()
        } catch {
            recordingService.stopRecording()
            activeSession = nil
            if let recordingError = error as? AudioRecordingService.RecordingError,
               case .permissionDenied = recordingError {
                shouldShowSettingsShortcut = true
            }
            let description = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            errorMessage = "Could not start recording: \(description)"
            isRecording = false
        }
    }

    func stopRecording() async {
        guard isRecording else { return }
        stopTimer()
        recordingService.stopRecording()
        isRecording = false

        guard let session = activeSession, let context = modelContext else { return }

        session.endedAt = Date()
        session.duration = max(elapsedTime, recordingService.currentDuration)

        do {
            try context.save()
            _ = try await flashbackService.generateHighlights(for: session, context: context)
        } catch {
            shouldShowSettingsShortcut = false
            errorMessage = "Recording saved, but highlight generation failed."
        }

        activeSession = nil
    }

    private func handleLaughter(offset: TimeInterval, confidence: Double) async {
        markers.append(LaughterMarker(offset: offset, confidence: confidence))

        guard let session = activeSession, let context = modelContext else { return }
        let event = LaughterEvent(timestampOffset: offset, confidence: confidence, session: session)
        session.laughterEvents.append(event)
        context.insert(event)

        do {
            try context.save()
        } catch {
            errorMessage = "Failed to save laughter marker."
        }
    }

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.elapsedTime = self.recordingService.currentDuration
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}
