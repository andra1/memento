import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, Flatten

# Modify feature extraction to return sequences instead of statistics
def extract_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, duration=3)  # Standardize to 3 seconds
    
    # Pad or truncate audio to standard length
    target_length = 3 * sr
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Combine features
    features = np.concatenate([mel_spec_db, mfcc], axis=0)
    features = features.T  # Reshape to (time_steps, features)
    
    return features

def prepare_dataset(laughter_files, non_laughter_files):
    X = []
    y = []
    
    # Process laughter samples
    for audio_file in laughter_files:
        features = extract_features(audio_file)
        X.append(features)
        y.append(1)
    
    # Process non-laughter samples
    for audio_file in non_laughter_files:
        features = extract_features(audio_file)
        X.append(features)
        y.append(0)
    
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential([
        # CNN layers
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        # LSTM layers
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def train_laughter_detector():
    # Replace these lists with your actual audio file paths
    laughter_files = ['path/to/laughter1.wav', 'path/to/laughter2.wav']
    non_laughter_files = ['path/to/non_laughter1.wav', 'path/to/non_laughter2.wav']
    
    # Prepare dataset
    X, y = prepare_dataset(laughter_files, non_laughter_files)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_laughter_detector()
