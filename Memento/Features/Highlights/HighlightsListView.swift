import SwiftData
import SwiftUI

struct HighlightsListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \HighlightClip.createdAt, order: .reverse)
    private var clips: [HighlightClip]

    @StateObject private var player = AudioPlaybackController()

    var body: some View {
        List {
            if clips.isEmpty {
                ContentUnavailableView(
                    "No Highlights",
                    systemImage: "waveform.badge.magnifyingglass",
                    description: Text("Your extracted laughter clips will appear here.")
                )
            } else {
                ForEach(clips) { clip in
                    clipRow(clip)
                }
                .onDelete(perform: delete)
            }
        }
        .navigationTitle("Highlights")
        .toolbar {
            EditButton()
        }
    }

    private func clipRow(_ clip: HighlightClip) -> some View {
        HStack(spacing: 12) {
            Button {
                player.togglePlayback(for: clip)
            } label: {
                Image(systemName: player.currentlyPlayingClipID == clip.id ? "stop.fill" : "play.fill")
                    .font(.title3)
            }
            .buttonStyle(.borderless)

            VStack(alignment: .leading, spacing: 4) {
                Text(clip.createdAt, style: .date)
                    .font(.subheadline.weight(.semibold))
                Text("\(clip.sourceStartOffset.formatted(.number.precision(.fractionLength(1))))s - \(clip.sourceEndOffset.formatted(.number.precision(.fractionLength(1))))s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                clip.isFavorited.toggle()
                try? modelContext.save()
            } label: {
                Image(systemName: clip.isFavorited ? "heart.fill" : "heart")
                    .foregroundStyle(clip.isFavorited ? .pink : .secondary)
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            let clip = clips[index]
            try? FileManager.default.removeItem(at: clip.clipFileURL)
            modelContext.delete(clip)
        }
        try? modelContext.save()
    }
}
