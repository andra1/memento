import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
#import torch
#import torchaudio
import numpy as np
from datetime import datetime
from laughter_detection import LaughterDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///laughterapp.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    highlights = db.relationship('Highlight', backref='user', lazy=True)

class Highlight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.Float, nullable=False)
    score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    highlights = Highlight.query.filter_by(user_id=current_user.id).order_by(Highlight.created_at.desc()).all()
    return render_template('dashboard.html', highlights=highlights)

@app.route('/upload', methods=['POST'])
@login_required
def upload_audio():
    if 'audio' not in request.files:
        app.logger.error("No audio file in request")
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        app.logger.error("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file extension is allowed
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        app.logger.error(f"Invalid file extension: {file_ext}")
        return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(allowed_extensions)}'}), 400
    
    try:
        # Secure the filename and create full path
        filename = secure_filename(audio_file.filename)
        app.logger.info(f"Processing file: {filename}")
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        app.logger.info(f"Saving to: {filepath}")
        
        # Save the uploaded file
        audio_file.save(filepath)
        app.logger.info(f"File saved successfully")
        
        # Process audio file and get highlights
        highlights = process_audio(filepath, current_user.id, filename)
        app.logger.info(f"Processed {len(highlights)} highlights")
        
        response_data = {
            'message': 'File uploaded and processed successfully',
            'filename': filename,
            'highlights': highlights
        }
        app.logger.info(f"Returning response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/check-files')
@login_required
def check_files():
    """Debug route to check uploaded files."""
    upload_dir = app.config['UPLOAD_FOLDER']
    files = []
    total_size = 0
    
    try:
        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            size = os.path.getsize(filepath)
            modified = os.path.getmtime(filepath)
            files.append({
                'name': filename,
                'size': f"{size/1024/1024:.2f} MB",
                'modified': datetime.fromtimestamp(modified)
            })
            total_size += size
            
        return render_template('check_files.html',
                             files=files,
                             total_size=f"{total_size/1024/1024:.2f} MB")
    except Exception as e:
        return f"Error checking files: {str(e)}"

@app.route('/check-database')
@login_required
def check_database():
    """Debug route to check database entries."""
    try:
        # Get all highlights for current user
        highlights = Highlight.query.filter_by(user_id=current_user.id)\
                                 .order_by(Highlight.created_at.desc())\
                                 .all()
                                 
        # Group highlights by filename
        files_dict = {}
        for h in highlights:
            if h.filename not in files_dict:
                files_dict[h.filename] = []
            files_dict[h.filename].append({
                'timestamp': h.timestamp,
                'score': h.score,
                'created_at': h.created_at
            })
            
        return render_template('check_database.html',
                             files_dict=files_dict,
                             total_highlights=len(highlights))
    except Exception as e:
        return f"Error checking database: {str(e)}"

def process_audio(filepath, user_id, filename):
    # Get laughter detection results
    detector = LaughterDetector()
    laughter_segments = detector.process_audio(filepath)
    results = [(segment.start_time, segment.score) for segment in laughter_segments]
    
    # Store highlights in database
    highlights = []
    for timestamp, score in results:
        highlight = Highlight(
            user_id=user_id,
            filename=filename,
            timestamp=timestamp,
            score=score
        )
        db.session.add(highlight)
        highlights.append({
            'timestamp': timestamp,
            'score': score
        })
    
    db.session.commit()
    return highlights

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 