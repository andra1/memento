import AVFoundation
import Foundation

@MainActor
final class AudioPlaybackController: NSObject, ObservableObject {
    @Published private(set) var currentlyPlayingClipID: UUID?

    private var player: AVAudioPlayer?

    func togglePlayback(for clip: HighlightClip) {
        if currentlyPlayingClipID == clip.id {
            stop()
            return
        }

        do {
            player = try AVAudioPlayer(contentsOf: clip.clipFileURL)
            player?.prepareToPlay()
            player?.play()
            currentlyPlayingClipID = clip.id
        } catch {
            currentlyPlayingClipID = nil
        }
    }

    func stop() {
        player?.stop()
        player = nil
        currentlyPlayingClipID = nil
    }
}
