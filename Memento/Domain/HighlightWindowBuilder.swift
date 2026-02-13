import Foundation

struct HighlightWindowBuilder {
    struct DetectionPoint {
        let offset: TimeInterval
        let confidence: Double

        init(offset: TimeInterval, confidence: Double = 1.0) {
            self.offset = offset
            self.confidence = confidence
        }
    }

    struct Configuration {
        var preRoll: TimeInterval = 3
        var postRoll: TimeInterval = 5
        var burstGap: TimeInterval = 1.0
        var mergeGap: TimeInterval = 0.75
        var minClipDuration: TimeInterval = 6
        var minConfidence: Double = 0.45
        var maxConfidencePostBoost: TimeInterval = 1.25
    }

    private struct Burst {
        var start: TimeInterval
        var end: TimeInterval
        var peakConfidence: Double
    }

    static func buildWindows(
        events: [DetectionPoint],
        sessionDuration: TimeInterval,
        configuration: Configuration = Configuration()
    ) -> [ClosedRange<TimeInterval>] {
        guard sessionDuration > 0 else { return [] }

        let detections = events
            .filter { point in
                point.offset.isFinite &&
                point.confidence.isFinite &&
                point.offset >= 0 &&
                point.offset <= sessionDuration &&
                point.confidence >= configuration.minConfidence
            }
            .sorted { $0.offset < $1.offset }

        guard let first = detections.first else { return [] }

        var bursts: [Burst] = []
        var current = Burst(start: first.offset, end: first.offset, peakConfidence: first.confidence)

        for point in detections.dropFirst() {
            if point.offset - current.end <= configuration.burstGap {
                current.end = point.offset
                current.peakConfidence = max(current.peakConfidence, point.confidence)
            } else {
                bursts.append(current)
                current = Burst(start: point.offset, end: point.offset, peakConfidence: point.confidence)
            }
        }

        bursts.append(current)

        let rawWindows = bursts.map { burst in
            let confidenceScale = normalizedConfidence(
                burst.peakConfidence,
                minimum: configuration.minConfidence
            )
            let boostedPostRoll = configuration.postRoll + (configuration.maxConfidencePostBoost * confidenceScale)

            var start = burst.start - configuration.preRoll
            var end = burst.end + boostedPostRoll

            let targetDuration = max(
                configuration.minClipDuration,
                configuration.preRoll + (burst.end - burst.start) + boostedPostRoll
            )

            rebalanceWindow(
                start: &start,
                end: &end,
                targetDuration: targetDuration,
                sessionDuration: sessionDuration
            )

            return start...end
        }

        return merge(windows: rawWindows, mergeGap: configuration.mergeGap)
    }

    static func buildWindows(
        eventOffsets: [TimeInterval],
        preRoll: TimeInterval,
        postRoll: TimeInterval,
        sessionDuration: TimeInterval,
        mergeGap: TimeInterval = 0.25
    ) -> [ClosedRange<TimeInterval>] {
        guard !eventOffsets.isEmpty, sessionDuration > 0 else { return [] }

        let sorted = eventOffsets.sorted()
        let windows = sorted.map { offset in
            max(0, offset - preRoll)...min(sessionDuration, offset + postRoll)
        }

        return merge(windows: windows, mergeGap: mergeGap)
    }

    private static func merge(
        windows: [ClosedRange<TimeInterval>],
        mergeGap: TimeInterval
    ) -> [ClosedRange<TimeInterval>] {
        guard var current = windows.first else { return [] }
        var merged: [ClosedRange<TimeInterval>] = []

        for window in windows.dropFirst() {
            if window.lowerBound <= current.upperBound + mergeGap {
                current = current.lowerBound...max(current.upperBound, window.upperBound)
            } else {
                merged.append(current)
                current = window
            }
        }

        merged.append(current)
        return merged
    }

    private static func normalizedConfidence(
        _ confidence: Double,
        minimum: Double
    ) -> Double {
        let denominator = max(0.001, 1 - minimum)
        let value = (confidence - minimum) / denominator
        return min(1, max(0, value))
    }

    private static func rebalanceWindow(
        start: inout TimeInterval,
        end: inout TimeInterval,
        targetDuration: TimeInterval,
        sessionDuration: TimeInterval
    ) {
        start = max(0, start)
        end = min(sessionDuration, end)

        if end < start {
            end = start
        }

        let desired = min(sessionDuration, max(0, targetDuration))
        let currentDuration = end - start
        guard currentDuration < desired else { return }

        let expansion = desired - currentDuration
        start -= expansion / 2
        end += expansion / 2

        if start < 0 {
            end = min(sessionDuration, end - start)
            start = 0
        }

        if end > sessionDuration {
            start = max(0, start - (end - sessionDuration))
            end = sessionDuration
        }
    }
}
