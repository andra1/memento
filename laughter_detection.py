import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import resampy
from pydub import AudioSegment
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

class InvalidAudioError(Exception):
    """Custom exception for invalid audio files."""
    pass

@dataclass
class LaughterSegment:
    start_time: float
    end_time: float
    score: float

    def __post_init__(self):
        """Validate segment data."""
        if self.start_time < 0:
            raise ValueError("start_time cannot be negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")

class LaughterDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Load YAMNet model
            self.logger.info("Loading YAMNet model...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YAMNet model: {str(e)}")
            raise ModelLoadError(f"Failed to load YAMNet model: {str(e)}")

        # YAMNet class mapping - index 505 corresponds to "Laughter"
        self.laughter_class_index = 505
        # Configuration
        self.chunk_duration = 0.96  # YAMNet's native segment length
        self.overlap_ratio = 0.5    # 50% overlap between segments
        self.min_score = 1e-2       # Threshold for detection
        self.sample_rate = 16000    # Required by YAMNet
        self.supported_formats = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        
    def validate_audio_file(self, file_path: str) -> None:
        """
        Validate the audio file exists and has a supported format.
        
        Args:
            file_path: Path to the audio file
            
        Raises:
            InvalidAudioError: If the file is invalid or unsupported
        """
        path = Path(file_path)
        if not path.exists():
            raise InvalidAudioError(f"Audio file not found: {file_path}")
        if path.suffix.lower() not in self.supported_formats:
            raise InvalidAudioError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
    def merge_overlapping_segments(self, segments: List[LaughterSegment]) -> List[LaughterSegment]:
        """
        Merge overlapping segments with similar scores.
        
        Args:
            segments: List of LaughterSegment objects
            
        Returns:
            List of merged LaughterSegment objects
        """
        if not segments:
            return []
        
        try:
            # Sort segments by start time
            segments.sort(key=lambda x: x.start_time)
            print(f"{len(segments)} segments before merging")
            
            merged = []
            current = segments[0]
            
            for next_seg in segments[1:]:
                # If segments overlap and have similar scores (within 0.2)
                if (next_seg.start_time <= current.end_time and 
                    abs(next_seg.score - current.score) < 1e-7):
                    # Merge segments
                    current = LaughterSegment(
                        start_time=current.start_time,
                        end_time=max(current.end_time, next_seg.end_time),
                        score=(current.score + next_seg.score) / 2
                    )
                    print(f"Merged segments: {current.start_time} - {current.end_time} (score: {current.score})")
                else:
                    merged.append(current)
                    current = next_seg
            
            merged.append(current)

            #Sort the merged segments by score
            print(f"{len(merged)} segments before sorting")
            merged.sort(key=lambda x: x.score, reverse=True)
            print(f"{len(merged)} segments after sorting")

            # If more than 5 segments, keep only top 5 by score
            if len(merged) > 5:
                highlights = merged[:5]
            else:
                highlights = merged[:len(merged) - 1]
            return highlights
            
        except Exception as e:
            self.logger.error(f"Error merging segments: {str(e)}")
            raise AudioProcessingError(f"Failed to merge segments: {str(e)}")
        
    def convert_to_wav(self, file_path: str) -> Tuple[str, bool]:
        """
        Convert audio file to WAV format if needed.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (wav_file_path, needs_cleanup)
        """
        try:
            # Debug: Print the file path
            self.logger.info(f"Attempting to convert file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                raise AudioProcessingError(f"File not found: {file_path}")

            if file_path.lower().endswith('.wav'):
                return file_path, False
            
            
            wav_path = f"{base_path}.wav"
            audio = AudioSegment.from_file(file_path, format="m4a")
            audio.export(wav_path, format='wav')
            print(f"Audio converted to WAV: {wav_path}")
            return wav_path, True
            
        except Exception as e:
            self.logger.error(f"Error converting audio to WAV: {str(e)}")
            raise AudioProcessingError(f"Failed to convert audio to WAV: {str(e)}")
        
    def process_audio(self, file_path: str) -> List[LaughterSegment]:
        """Process audio using YAMNet's optimal segment size."""
        try:
            # Load and resample audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Calculate segment sizes
            segment_samples = int(self.chunk_duration * self.sample_rate)
            hop_samples = int(segment_samples * (1 - self.overlap_ratio))
            
            segments = []
            for start_sample in range(0, len(audio) - segment_samples + 1, hop_samples):
                end_sample = start_sample + segment_samples
                print(start_sample, end_sample)
                chunk = audio[start_sample:end_sample]
                
                # Get predictions
                scores, embeddings, spectrogram = self.model(chunk)
                scores = scores.numpy()
                
                # Get laughter score for this segment
                laughter_score = float(np.mean(scores[:, self.laughter_class_index]))

                print(laughter_score)
                # Store all segments with their scores
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                segments.append(LaughterSegment(
                    start_time=start_time,
                    end_time=end_time, 
                    score=laughter_score
                ))
                self.logger.info(f"Segment analyzed: {start_time:.2f}s - {end_time:.2f}s (score: {laughter_score:.4f})")
                
                # 
            return self.merge_overlapping_segments(segments)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            raise AudioProcessingError(f"Failed to process audio: {str(e)}")

    def get_chunk_predictions(self, audio_chunk: np.ndarray, sr: int) -> float:
        """
        Get predictions for a single audio chunk.
        
        Args:
            audio_chunk: Audio waveform
            sr: Sample rate
            
        Returns:
            float: Laughter score for the chunk
            
        Raises:
            ValueError: If input parameters are invalid
            AudioProcessingError: If processing fails
        """
        try:
            # Validate inputs
            if not isinstance(audio_chunk, np.ndarray):
                raise ValueError("audio_chunk must be a numpy array")
            if not isinstance(sr, int) or sr <= 0:
                raise ValueError("sr must be a positive integer")
            if len(audio_chunk) == 0:
                raise ValueError("audio_chunk is empty")
            
            # Ensure audio is at 16kHz
            if sr != 16000:
                try:
                    audio_chunk = resampy.resample(audio_chunk, sr, 16000)
                except Exception as e:
                    raise AudioProcessingError(f"Failed to resample audio chunk: {str(e)}")
            
            # Get model predictions
            try:
                scores, embeddings, spectrogram = self.model(audio_chunk)
                scores = scores.numpy()
            except Exception as e:
                raise AudioProcessingError(f"Model prediction failed: {str(e)}")
            
            # Get average laughter score
            laughter_scores = scores[:, self.laughter_class_index]
            return float(np.mean(laughter_scores))
            
        except Exception as e:
            self.logger.error(f"Error in get_chunk_predictions: {str(e)}")
            raise

    def analyze_audio_events(self, audio_chunk: np.ndarray) -> None:
        """
        Analyze audio chunk and print the most confident audio event class.
        
        Args:
            audio_chunk: Audio waveform as numpy array at 16kHz sample rate
        """
        try:
            # Get model predictions
            scores, embeddings, spectrogram = self.model(audio_chunk)
            scores = scores.numpy()
            
            # Get class names from YAMNet (fixed method)
            class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
                                                    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv')
            class_names = []
            with open(class_map_path) as f:
                # Skip header row
                next(f)
                for row in f:
                    # Split on comma and get the display name (third column)
                    parts = row.strip().split(',')
                    if len(parts) >= 3:  # Ensure we have enough columns
                        class_names.append(parts[2])
            
            # Find the highest scoring class
            top_class_idx = scores.mean(axis=0).argmax()
            top_class_name = class_names[top_class_idx]
            top_score = float(scores[:, top_class_idx].mean())
            
            print(f"Top audio event: {top_class_name} (score: {top_score:.4f})")
            
            # Optional: Print top 5 classes
            top_5_idx = scores.mean(axis=0).argsort()[-5:][::-1]
            print("\nTop 5 audio events:")
            for idx in top_5_idx:
                class_name = class_names[idx]
                score = float(scores[:, idx].mean())
                print(f"{class_name}: {score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio events: {str(e)}")
            raise AudioProcessingError(f"Failed to analyze audio events: {str(e)}")

    def process_audio_events(self, file_path: str) -> None:
        """Process entire audio file in optimal chunks and analyze events."""
        try:
            # Load and resample audio
            audio, sr = librosa.load(file_path, 
                               sr=self.sample_rate,  # 16000 Hz
                               mono=True)            # Ensure mono
        
            # Calculate segment size for 0.96 seconds
            segment_samples = int(self.chunk_duration * self.sample_rate)  # 0.96 * 16000 = 15360 samples
        
            print(f"\nAnalyzing audio file: {file_path}")
            print(f"Sample rate: {sr} Hz")
            print(f"Duration: {len(audio)/sr:.2f} seconds")
            print(f"Chunk size: {self.chunk_duration} seconds ({segment_samples} samples)")
            print("--------------------------------")
        
            # Process each chunk
            for start_sample in range(0, len(audio) - segment_samples + 1, segment_samples):
                end_sample = start_sample + segment_samples
                chunk = audio[start_sample:end_sample]
            
                # Ensure chunk is the right size
                if len(chunk) == segment_samples:
                    # Print timestamp
                    start_time = start_sample / self.sample_rate
                    print(f"\nTimestamp: {start_time:.2f}s")
                
                    # Normalize audio chunk if needed
                    if np.abs(chunk).max() > 1.0:
                        chunk = chunk / np.abs(chunk).max()
                
                    self.analyze_audio_events(chunk)
            
        except Exception as e:
            self.logger.error(f"Error processing audio events: {str(e)}")
            raise AudioProcessingError(f"Failed to process audio events: {str(e)}")

    def print_top_classes(self, n: int = 10) -> None:
        """
        Print the top n most common audio classes from a sample prediction.
        
        Args:
            n: Number of top classes to print
        """
        try:
            class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
                                                    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv')
            print("\nTop YAMNet audio classes:")
            print("------------------------")
            with open(class_map_path) as f:
                # Skip header if present
                if 'display_name' in f.readline():
                    pass
                # Print first n classes
                for i, row in enumerate(f):
                    if i >= n:
                        break
                    parts = row.strip().split(',')
                    # The display name is always the third column
                    class_name = parts[2]
                    print(f"{i}: {class_name}")
                
        except Exception as e:
            self.logger.error(f"Error printing classes: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Create a singleton instance
        detector = LaughterDetector()
        results = detector.process_audio("files\shraya_test.wav")
        #detector.process_audio_events("files\laughter.wav")
        print(results)
       
    except ModelLoadError as e:
        logger.critical(f"Failed to initialize LaughterDetector: {str(e)}")
        raise