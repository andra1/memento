# Memento

Memento is an iOS SwiftUI + SwiftData app that records conversations, detects laughter in real time, extracts highlight clips, and surfaces them as next-day flashbacks.

## MVP features implemented

- Sign in with Apple auth gate
- Conversation recording with `AVAudioEngine`
- Real-time laughter detection with `SoundAnalysis`
- Automatic clip extraction around laughter events
- SwiftData persistence for sessions, events, clips, and user settings
- Daily flashback generation orchestration
- Highlights list with playback, favorite, and delete
- Notification toggle for daily flashbacks

## Build

1. Generate the project:

```bash
xcodegen generate
```

2. Open `Memento.xcodeproj` in Xcode and run on an iOS simulator/device.

## Notes

- Audio processing and clip generation are currently on-device.
- Background continuous recording is not included in this MVP.
