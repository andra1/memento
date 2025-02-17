# Laughter Detection Web Application

A web application that detects and highlights moments of laughter in audio files using machine learning.

## Features

- User authentication system
- Audio file upload and processing
- Laughter detection using YAMNet model
- Top 5 laughter highlights extraction
- Audio playback of highlighted segments
- Modern, responsive UI

## Technical Stack

- **Backend**: Flask, SQLAlchemy
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **ML Model**: TensorFlow YAMNet
- **Audio Processing**: librosa, pydub
- **Database**: SQLite

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd laughter-detection-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Register a new account or login
2. Upload an audio file (supported formats: WAV, MP3, OGG, FLAC, M4A)
3. Wait for the processing to complete
4. View and play the detected laughter highlights

## Project Structure

- `app.py`: Main Flask application
- `laughter_detection.py`: Laughter detection module
- `templates/`: HTML templates
- `uploads/`: Directory for uploaded audio files (created automatically)
- `requirements.txt`: Python dependencies

## Contributing

Feel free to submit issues and enhancement requests! 