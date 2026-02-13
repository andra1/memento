import XCTest
@testable import Memento

final class HighlightWindowBuilderTests: XCTestCase {
    private let accuracy: TimeInterval = 0.001

    func testMergesNearbyWindows() {
        let windows = HighlightWindowBuilder.buildWindows(
            eventOffsets: [10, 12],
            preRoll: 3,
            postRoll: 5,
            sessionDuration: 60
        )

        XCTAssertEqual(windows.count, 1)
        guard let first = windows.first else {
            XCTFail("Expected one merged window")
            return
        }
        XCTAssertEqual(first.lowerBound, 7, accuracy: accuracy)
        XCTAssertEqual(first.upperBound, 17, accuracy: accuracy)
    }

    func testClampsWindowsToSessionBounds() {
        let windows = HighlightWindowBuilder.buildWindows(
            eventOffsets: [1, 29],
            preRoll: 5,
            postRoll: 5,
            sessionDuration: 30
        )

        XCTAssertEqual(windows.count, 2)
        XCTAssertEqual(windows[0].lowerBound, 0, accuracy: accuracy)
        XCTAssertEqual(windows[0].upperBound, 6, accuracy: accuracy)
        XCTAssertEqual(windows[1].lowerBound, 24, accuracy: accuracy)
        XCTAssertEqual(windows[1].upperBound, 30, accuracy: accuracy)
    }

    func testGroupsDenseDetectionsIntoSingleBurstWindow() {
        let configuration = HighlightWindowBuilder.Configuration(
            preRoll: 2,
            postRoll: 4,
            burstGap: 1,
            mergeGap: 0.75,
            minClipDuration: 0,
            minConfidence: 0.45,
            maxConfidencePostBoost: 0
        )

        let windows = HighlightWindowBuilder.buildWindows(
            events: [
                .init(offset: 10, confidence: 0.6),
                .init(offset: 10.4, confidence: 0.8),
                .init(offset: 10.9, confidence: 0.7)
            ],
            sessionDuration: 40,
            configuration: configuration
        )

        XCTAssertEqual(windows.count, 1)
        XCTAssertEqual(windows[0].lowerBound, 8, accuracy: accuracy)
        XCTAssertEqual(windows[0].upperBound, 14.9, accuracy: accuracy)
    }

    func testRebalancesStartBoundedWindowToPreserveMinimumDuration() {
        let configuration = HighlightWindowBuilder.Configuration(
            preRoll: 3,
            postRoll: 4,
            burstGap: 1,
            mergeGap: 0.75,
            minClipDuration: 8,
            minConfidence: 0.45,
            maxConfidencePostBoost: 0
        )

        let windows = HighlightWindowBuilder.buildWindows(
            events: [.init(offset: 0.4, confidence: 0.7)],
            sessionDuration: 10,
            configuration: configuration
        )

        XCTAssertEqual(windows.count, 1)
        XCTAssertEqual(windows[0].lowerBound, 0, accuracy: accuracy)
        XCTAssertEqual(windows[0].upperBound, 8, accuracy: accuracy)
    }

    func testRebalancesEndBoundedWindowToPreserveMinimumDuration() {
        let configuration = HighlightWindowBuilder.Configuration(
            preRoll: 3,
            postRoll: 4,
            burstGap: 1,
            mergeGap: 0.75,
            minClipDuration: 8,
            minConfidence: 0.45,
            maxConfidencePostBoost: 0
        )

        let windows = HighlightWindowBuilder.buildWindows(
            events: [.init(offset: 9.6, confidence: 0.7)],
            sessionDuration: 10,
            configuration: configuration
        )

        XCTAssertEqual(windows.count, 1)
        XCTAssertEqual(windows[0].lowerBound, 2, accuracy: accuracy)
        XCTAssertEqual(windows[0].upperBound, 10, accuracy: accuracy)
    }

    func testConfidenceBoostExtendsPostRoll() {
        let configuration = HighlightWindowBuilder.Configuration(
            preRoll: 2,
            postRoll: 3,
            burstGap: 1,
            mergeGap: 0.75,
            minClipDuration: 0,
            minConfidence: 0.45,
            maxConfidencePostBoost: 2
        )

        let windows = HighlightWindowBuilder.buildWindows(
            events: [.init(offset: 10, confidence: 1)],
            sessionDuration: 30,
            configuration: configuration
        )

        XCTAssertEqual(windows.count, 1)
        XCTAssertEqual(windows[0].lowerBound, 8, accuracy: accuracy)
        XCTAssertEqual(windows[0].upperBound, 15, accuracy: accuracy)
    }
}
