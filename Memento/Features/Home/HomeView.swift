import SwiftData
import SwiftUI

struct HomeView: View {
    @Query(sort: \HighlightClip.createdAt, order: .reverse)
    private var clips: [HighlightClip]

    private var yesterdayClips: [HighlightClip] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today) ?? today
        return clips.filter { calendar.isDate($0.dayBucket, inSameDayAs: yesterday) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Today")
                    .font(.largeTitle.bold())

                flashbackCard

                Text("Recent Highlights")
                    .font(.headline)
                    .padding(.top, 8)

                if clips.isEmpty {
                    ContentUnavailableView(
                        "No Highlights Yet",
                        systemImage: "sparkles",
                        description: Text("Record a conversation to generate laughter flashbacks.")
                    )
                } else {
                    ForEach(clips.prefix(10)) { clip in
                        HighlightSummaryRow(clip: clip)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Memento")
    }

    private var flashbackCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Yesterday's Flashback")
                .font(.headline)

            if yesterdayClips.isEmpty {
                Text("No clips from yesterday yet.")
                    .foregroundStyle(.secondary)
            } else {
                Text("\(yesterdayClips.count) highlights are ready.")
                    .font(.title3.weight(.semibold))
                Text("Open Highlights to replay your best moments.")
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Color.cyan.opacity(0.25), Color.blue.opacity(0.18)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
    }
}

private struct HighlightSummaryRow: View {
    let clip: HighlightClip

    var body: some View {
        HStack {
            Image(systemName: "waveform")
                .foregroundStyle(.blue)

            VStack(alignment: .leading) {
                Text(clip.createdAt, style: .time)
                    .font(.subheadline.weight(.semibold))
                Text("\(clip.sourceStartOffset.formatted(.number.precision(.fractionLength(1))))s - \(clip.sourceEndOffset.formatted(.number.precision(.fractionLength(1))))s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if clip.isFavorited {
                Image(systemName: "heart.fill")
                    .foregroundStyle(.pink)
            }
        }
        .padding(.vertical, 6)
    }
}
