"""
Stuttering Disfluency Detection Backend API
Main Flask application with model inference and data management
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torchaudio
import traceback

# Import local modules
from models import AcousticModel, MultimodalModel
from model_service import ModelInferenceService
from database import Database
from config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Setup directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Initialize services
db = Database(app.config['DATABASE_PATH'])
inference_service = None

def init_models():
    """Initialize model inference service"""
    global inference_service
    try:
        inference_service = ModelInferenceService(app.config['MODELS_FOLDER'])
        print("✓ Models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing models: {str(e)}")
        return False

@app.before_request
def before_request():
    """Initialize models on first request"""
    global inference_service
    if inference_service is None:
        init_models()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': inference_service.models_loaded if inference_service else False,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """
    Analyze audio file for disfluencies
    Accepts: audio file upload or voice recording
    Returns: Prediction results with timestamps and confidence scores
    """
    try:
        print("=== DEBUG REQUEST ===")
        print("FILES:", request.files)
        print("FORM:", request.form)
        print("audio:", request.files.get('audio'))
        print("file:", request.files.get('file'))
        print("modality:", request.form.get('modality'))
        print("====================")

        # Check if file is in request
        uploaded_file = request.files.get('audio') or request.files.get('file')

        if uploaded_file is None:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = uploaded_file

        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get analysis parameters
        modality = request.form.get('modality', 'multimodal')
        user_id = request.form.get('user_id', 'anonymous')
        notes = request.form.get('notes', '')

        # Validate modality
        if modality not in ['language', 'acoustic', 'multimodal']:
            return jsonify({
                'error': f'Invalid modality: {modality}. Must be: language, acoustic, or multimodal'
            }), 400
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        print(f"Processing audio: {filename}")
        
        # Run analysis
        results = inference_service.analyze(filepath, modality)

        # Ensure signal-level features included (fallback if inference service didn't attach)
        try:
            if results.get('success'):
                # attach features if missing
                if 'signal_features' not in results.get('data', {}):
                    try:
                        features = inference_service._extract_audio_features(filepath)
                    except Exception:
                        features = {}
                    results['data']['signal_features'] = features
        except Exception:
            pass
        
        if results['success']:
            # Save to database
            analysis_id = db.save_analysis(
                user_id=user_id,
                filename=filename,
                filepath=filepath,
                modality=modality,
                results=results['data'],
                notes=notes
            )
            
            # Prepare response
            response_data = {
                'success': True,
                'analysis_id': analysis_id,
                'filename': filename,
                'modality': modality,
                'timestamp': datetime.now().isoformat(),
                'results': results['data']
            }
            
            return jsonify(response_data), 200
        else:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 400
            
    except Exception as e:
        print(f"Error in analyze_audio: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e)
        }), 500

@app.route('/api/results/<analysis_id>', methods=['GET'])
def get_results(analysis_id):
    """Get saved analysis results by ID"""
    try:
        result = db.get_analysis(analysis_id)
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/user/<user_id>', methods=['GET'])
def get_user_results(user_id):
    """Get all analyses for a specific user"""
    try:
        results = db.get_user_analyses(user_id)
        return jsonify({
            'user_id': user_id,
            'total_analyses': len(results),
            'results': results
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def list_results():
    """List all analyses with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        user_id = request.args.get('user_id', type=str)
        if user_id:
            user_id = user_id.strip() or None
        
        results = db.get_all_analyses(page=page, per_page=per_page, user_id=user_id)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<analysis_id>', methods=['DELETE'])
def delete_result(analysis_id):
    """Delete an analysis result"""
    try:
        if db.delete_analysis(analysis_id):
            return jsonify({'success': True, 'message': 'Analysis deleted'}), 200
        else:
            return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<analysis_id>', methods=['GET'])
def export_results(analysis_id):
    """Export analysis results as CSV"""
    try:
        result = db.get_analysis(analysis_id)
        if not result:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Create CSV from results
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.csv")
        
        # Convert results to DataFrame and save
        if 'predictions' in result['results']:
            df = pd.DataFrame(result['results']['predictions'])
            df.to_csv(csv_path, index=False)
            
            return send_file(csv_path, as_attachment=True, 
                           download_name=f"analysis_{analysis_id}.csv")
        else:
            return jsonify({'error': 'No predictions to export'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        stats = db.get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models and their info"""
    try:
        if inference_service is None:
            return jsonify({'error': 'Models not loaded'}), 503
        
        models_info = inference_service.get_available_models()
        return jsonify(models_info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def serve_index():
    """Serve index.html"""
    try:
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
        return send_file(index_path, mimetype='text/html')
    except Exception as e:
        print(f"Error serving index: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Stuttering Disfluency Detection API")
    print("=" * 60)
    
    # Initialize database
    db.init_db()
    print("✓ Database initialized")
    
    # Start Flask app
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
