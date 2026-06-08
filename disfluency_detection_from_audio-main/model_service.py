"""
Model Inference Service
Handles model loading and prediction for audio analysis
"""
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from models import AcousticModel, MultimodalModel

# Labels for different disfluency types
LABELS = ['FP', 'RP', 'RV', 'RS', 'PW']
LABEL_NAMES = {
    'FP': 'Filled Pause',
    'RP': 'Partial Repetition',
    'RV': 'Revision',
    'RS': 'Restart',
    'PW': 'Prolonged Word'
}

class ModelInferenceService:
    """Service for running inference on audio files"""
    
    def __init__(self, models_path):
        """
        Initialize the inference service
        Args:
            models_path: Path to directory containing pre-trained models
        """
        self.models_path = models_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.acoustic_model = None
        self.multimodal_model = None
        self._demo_module = None
        self._demo_import_error = None
        self.models_loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load acoustic model
            acoustic_path = os.path.join(self.models_path, 'acoustic.pt')
            if os.path.exists(acoustic_path):
                self.acoustic_model = AcousticModel()
                self.acoustic_model.load_state_dict(
                    torch.load(acoustic_path, map_location=self.device),
                    strict=False
                )
                self.acoustic_model = self.acoustic_model.to(self.device)
                self.acoustic_model.eval()
                print("Acoustic model loaded")
            else:
                print(f"Acoustic model not found at {acoustic_path}")
            
            # Load multimodal model
            multimodal_path = os.path.join(self.models_path, 'multimodal.pt')
            if os.path.exists(multimodal_path):
                self.multimodal_model = MultimodalModel()
                self.multimodal_model.load_state_dict(
                    torch.load(multimodal_path, map_location=self.device),
                    strict=False
                )
                self.multimodal_model = self.multimodal_model.to(self.device)
                self.multimodal_model.eval()
                print("Multimodal model loaded")
            else:
                print(f"Multimodal model not found at {multimodal_path}")
            
            self.models_loaded = self.acoustic_model is not None or self.multimodal_model is not None
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def _get_demo_module(self):
        """Lazily import the demo pipeline used for language analysis."""
        if self._demo_module is not None:
            return self._demo_module

        try:
            import demo
            self._demo_module = demo
            return demo
        except Exception as e:
            self._demo_import_error = str(e)
            return None

    def _resample_sequence(self, sequence, target_length):
        """Downsample or pad a sequence to the desired frame count."""
        if sequence is None:
            return None

        array = sequence.cpu().numpy() if hasattr(sequence, "cpu") else np.asarray(sequence)
        if array.ndim == 1:
            array = array[:, None]

        current_length = array.shape[0]
        if current_length == target_length:
            return array

        if current_length == 0:
            return np.zeros((target_length, array.shape[1]), dtype=array.dtype)

        if current_length > target_length:
            indices = np.linspace(0, current_length - 1, num=target_length).round().astype(int)
            return array[indices]

        padding = np.repeat(array[-1:, ...], target_length - current_length, axis=0)
        return np.concatenate([array, padding], axis=0)

    def _count_disfluency_events(self, frame_activity):
        """Count contiguous disfluency spans from a boolean activity vector."""
        if frame_activity.size == 0:
            return 0

        starts = np.logical_and(frame_activity, np.logical_not(np.concatenate(([False], frame_activity[:-1]))))
        return int(starts.sum())
    
    def analyze(self, audio_path, modality='multimodal'):
        """
        Analyze audio for disfluencies
        Args:
            audio_path: Path to audio file
            modality: Type of analysis ('acoustic', 'language', or 'multimodal')
        
        Returns:
            Dictionary with analysis results
        """
        try:
            if not os.path.exists(audio_path):
                return {'success': False, 'error': 'Audio file not found'}

            # Load and preprocess audio using librosa (more compatible than torchaudio)
            try:
                import librosa
                audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
                audio = torch.FloatTensor(audio_np).unsqueeze(0)
            except Exception as e:
                print(f"Librosa failed: {e}, trying torchaudio...")
                try:
                    audio, sr = torchaudio.load(audio_path)
                    
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        audio = resampler(audio)
                    
                    # Convert to mono if stereo
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                except Exception as e2:
                    return {'success': False, 'error': f'Failed to load audio: {str(e2)}'}
            
            audio = audio.to(self.device)
            
            print(f"[DEBUG] Audio shape: {audio.shape}")
            print(f"[DEBUG] Running {modality} analysis...")
            
            # Extract signal-level features (MFCC, pitch, energy, pauses, repetitions, prolongations)
            try:
                signal_features = self._extract_audio_features(audio_path)
            except Exception as feat_e:
                print(f"Warning: feature extraction failed: {feat_e}")
                signal_features = {}

            # Run analysis based on modality
            if modality == 'acoustic':
                if self.acoustic_model is None:
                    return {'success': False, 'error': 'Acoustic model not loaded'}
                print("[DEBUG] Calling _analyze_acoustic...")
                result = self._analyze_acoustic(audio, audio_path)
                if result.get('success'):
                    result['data']['signal_features'] = signal_features
                return result
            elif modality == 'language':
                print("[DEBUG] Calling _analyze_language...")
                result = self._analyze_language(audio_path)
                if result.get('success'):
                    result['data']['signal_features'] = signal_features
                return result
            elif modality == 'multimodal':
                if self.acoustic_model is None or self.multimodal_model is None:
                    return {'success': False, 'error': 'Required models not loaded'}
                print("[DEBUG] Calling _analyze_multimodal...")
                result = self._analyze_multimodal(audio, audio_path)
                if result.get('success'):
                    result['data']['signal_features'] = signal_features
                return result
            else:
                return {'success': False, 'error': 'Invalid modality'}
        
        except Exception as e:
            print(f"[ERROR] Error in analyze: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _analyze_acoustic(self, audio, audio_path):
        """Acoustic-based analysis"""
        try:
            print("[DEBUG] _analyze_acoustic: Starting")
            if self.acoustic_model is None:
                return {'success': False, 'error': 'Acoustic model not loaded'}
            
            print("[DEBUG] _analyze_acoustic: Model loaded, processing audio...")
            with torch.no_grad():
                # Get embeddings and predictions
                audio_input = audio[0].unsqueeze(0)
                print(f"[DEBUG] _analyze_acoustic: audio_input shape = {audio_input.shape}")
                embeddings, logits = self.acoustic_model(audio_input)
                print(f"[DEBUG] _analyze_acoustic: Got embeddings shape = {embeddings.shape}, logits shape = {logits.shape}")
                
                # Convert to probabilities
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.75).int()
                print(f"[DEBUG] _analyze_acoustic: predictions shape = {predictions.shape}")
                
                # Generate frame-level predictions
                frame_preds = self._generate_frame_predictions(
                    predictions, audio_path, modality='acoustic', probs=probs
                )
                
                # Calculate statistics
                stats = self._calculate_statistics(probs, predictions, audio_path)
                
                return {
                    'success': True,
                    'data': {
                        'modality': 'acoustic',
                        'predictions': frame_preds,
                        'statistics': stats,
                        'disfluency_count': stats['disfluency_events'],
                        'confidence': float(probs.max().item())
                    }
                }
        
        except Exception as e:
            print(f"[ERROR] _analyze_acoustic: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _analyze_language(self, audio_path):
        """Language-based analysis using ASR + BERT token classification."""
        try:
            demo = self._get_demo_module()
            if demo is None:
                return {
                    'success': False,
                    'error': f'Language pipeline unavailable: {self._demo_import_error}'
                }

            print("[DEBUG] _analyze_language: Starting")
            text_df = demo.run_asr(audio_path, self.device)
            frame_emb, frame_pred = demo.run_language_based(audio_path, text_df, self.device)

            if frame_pred.numel() == 0:
                return {'success': False, 'error': 'Language model produced no frame-level predictions'}

            probs = frame_pred.float()
            predictions = frame_pred.int()
            frame_preds = self._generate_frame_predictions(
                predictions, audio_path, modality='language', probs=probs
            )
            stats = self._calculate_statistics(probs, predictions, audio_path)

            return {
                'success': True,
                'data': {
                    'modality': 'language',
                    'predictions': frame_preds,
                    'statistics': stats,
                    'disfluency_count': stats['disfluency_events'],
                    'confidence': float(probs.max().item()) if probs.numel() > 0 else 0.0,
                    'transcript': text_df.to_dict(orient='records'),
                    'language_embedding_shape': list(frame_emb.shape)
                }
            }

        except Exception as e:
            print(f"[ERROR] _analyze_language: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _analyze_multimodal(self, audio, audio_path):
        """Multimodal analysis (acoustic + language features)"""
        try:
            print("[DEBUG] _analyze_multimodal: Starting")
            if self.acoustic_model is None or self.multimodal_model is None:
                return {'success': False, 'error': 'Required models not loaded'}
            
            print("[DEBUG] _analyze_multimodal: Models loaded")
            demo = self._get_demo_module()
            if demo is None:
                return {
                    'success': False,
                    'error': f'Language pipeline unavailable: {self._demo_import_error}'
                }

            with torch.no_grad():
                text_df = demo.run_asr(audio_path, self.device)
                language_emb, _ = demo.run_language_based(audio_path, text_df, self.device)

                # Get acoustic embeddings
                audio_input = audio[0].unsqueeze(0)
                print(f"[DEBUG] audio_input shape: {audio_input.shape}")
                acoustic_emb, acoustic_logits = self.acoustic_model(audio_input)
                print(f"[DEBUG] acoustic_emb shape: {acoustic_emb.shape}")
                acoustic_probs = torch.sigmoid(acoustic_logits)

                acoustic_emb = acoustic_emb[0]
                min_size = min(language_emb.size(0), acoustic_emb.size(0))
                if min_size == 0:
                    return {'success': False, 'error': 'Unable to align language and acoustic embeddings'}
                language_emb = language_emb[:min_size].unsqueeze(0)
                acoustic_emb = acoustic_emb[:min_size].unsqueeze(0)
                
                # Get multimodal predictions
                print(f"[DEBUG] Calling multimodal model...")
                multimodal_logits = self.multimodal_model(language_emb, acoustic_emb)
                print(f"[DEBUG] multimodal_logits shape: {multimodal_logits.shape}")
                multimodal_probs = torch.sigmoid(multimodal_logits)
                multimodal_preds = (multimodal_probs > 0.75).int()
                
                # Generate predictions
                frame_preds = self._generate_frame_predictions(
                    multimodal_preds, audio_path, modality='multimodal', probs=multimodal_probs
                )
                
                # Calculate statistics
                stats = self._calculate_statistics(multimodal_probs, multimodal_preds, audio_path)
                
                return {
                    'success': True,
                    'data': {
                        'modality': 'multimodal',
                        'predictions': frame_preds,
                        'statistics': stats,
                        'disfluency_count': stats['disfluency_events'],
                        'confidence': float(multimodal_probs.max().item()),
                        'acoustic_confidence': float(acoustic_probs.max().item()),
                        'transcript': text_df.to_dict(orient='records')
                    }
                }
        
        except Exception as e:
            print(f"[ERROR] _analyze_multimodal: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _generate_frame_predictions(self, predictions, audio_path, modality='acoustic', probs=None):
        """
        Generate frame-level predictions
        Frame size: 20ms (standard for audio analysis)
        """
        try:
            # Get audio duration using librosa
            try:
                import librosa
                audio_np, sr = librosa.load(audio_path, sr=None, mono=True)
                duration = len(audio_np) / sr
            except Exception:
                # Fallback: calculate from predictions shape
                # Assuming model processes 20ms frames
                duration = predictions.shape[1] * 0.02
            
            # Frame size in seconds (20ms)
            frame_size = 0.02
            num_frames = int(np.ceil(duration / frame_size))
            
            # Convert predictions to numpy and align them to the audio timeline
            preds_np = predictions.cpu().numpy()
            probs_np = probs.cpu().numpy() if probs is not None else None

            if preds_np.ndim == 3:
                frame_preds_np = preds_np[0]
            elif preds_np.ndim == 2:
                frame_preds_np = preds_np
            else:
                frame_preds_np = preds_np.reshape(-1, 1)

            if probs_np is not None:
                if probs_np.ndim == 3:
                    frame_probs_np = probs_np[0]
                elif probs_np.ndim == 2:
                    frame_probs_np = probs_np
                else:
                    frame_probs_np = probs_np.reshape(-1, 1)
            else:
                frame_probs_np = None

            frame_preds_np = self._resample_sequence(frame_preds_np, num_frames)
            if frame_probs_np is not None:
                frame_probs_np = self._resample_sequence(frame_probs_np, num_frames)
            
            # Create frame predictions
            frame_predictions = []
            for frame_idx in range(num_frames):
                start_time = frame_idx * frame_size
                end_time = (frame_idx + 1) * frame_size
                
                # Get prediction for this frame
                frame_pred = frame_preds_np[frame_idx] if frame_idx < frame_preds_np.shape[0] else np.zeros(len(LABELS), dtype=int)
                frame_prob = frame_probs_np[frame_idx] if frame_probs_np is not None and frame_idx < frame_probs_np.shape[0] else frame_pred
                
                # Determine if disfluent and which type
                is_disfluent = bool(np.max(frame_prob) > 0.75)
                disfluency_types = []

                for i in range(len(frame_prob)):
                    if frame_prob[i] > 0.75:
                        disfluency_types.append(LABEL_NAMES[LABELS[i]])
                
                frame_predictions.append({
                    'frame': frame_idx,
                    'start_time': round(start_time, 3),
                    'end_time': round(end_time, 3),
                    'is_disfluent': bool(is_disfluent),
                    'disfluency_types': disfluency_types,
                    'confidence': round(float(np.max(frame_prob)), 3) if is_disfluent else 0.0
                })
            
            return frame_predictions
        
        except Exception as e:
            print(f"Error generating frame predictions: {str(e)}")
            return []
    
    def _calculate_statistics(self, probs, predictions, audio_path=None):
        """Calculate statistics from predictions"""
        try:
            probs_np = probs.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            if preds_np.ndim == 3:
                frame_preds_np = preds_np[0]
            elif preds_np.ndim == 2:
                frame_preds_np = preds_np
            else:
                frame_preds_np = preds_np.reshape(-1, 1)

            if probs_np.ndim == 3:
                frame_probs_np = probs_np[0]
            elif probs_np.ndim == 2:
                frame_probs_np = probs_np
            else:
                frame_probs_np = probs_np.reshape(-1, 1)

            if audio_path:
                try:
                    import librosa
                    audio_np, sr = librosa.load(audio_path, sr=None, mono=True)
                    target_frames = int(np.ceil((len(audio_np) / sr) / 0.02))
                except Exception:
                    target_frames = frame_preds_np.shape[0]
            else:
                target_frames = frame_preds_np.shape[0]

            frame_preds_np = self._resample_sequence(frame_preds_np, target_frames)
            frame_probs_np = self._resample_sequence(frame_probs_np, target_frames)

            frame_activity = np.any(frame_preds_np > 0, axis=1)
            total_frames = int(frame_preds_np.shape[0])
            disfluent_frames = int(frame_activity.sum())
            disfluency_events = self._count_disfluency_events(frame_activity)

            stats = {
                'total_frames': int(total_frames),
                'disfluent_frames': disfluent_frames,
                'fluent_frames': int(total_frames - disfluent_frames),
                'disfluency_rate': round(float(disfluent_frames / total_frames * 100), 2) if total_frames > 0 else 0.0,
                'disfluency_events': disfluency_events,
            }
            
            # Per-label statistics
            for i, label in enumerate(LABELS):
                label_preds = frame_preds_np[:, i] if frame_preds_np.ndim > 1 else frame_preds_np
                label_prob = float(np.mean(frame_probs_np[:, i] if frame_probs_np.ndim > 1 else frame_probs_np))
                
                stats[f'{label}_count'] = int(np.sum(label_preds))
                stats[f'{label}_avg_confidence'] = round(label_prob, 3)
            
            return stats
        
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return {}
    
    def get_available_models(self):
        """Get information about available models"""
        language_assets = [
            Path(self.models_path) / 'language.pt',
            Path(self.models_path) / 'asr' / 'config.json',
            Path(self.models_path) / 'asr' / 'pytorch_model.bin',
        ]
        return {
            'available_models': {
                'acoustic': self.acoustic_model is not None,
                'multimodal': self.multimodal_model is not None,
                'language': all(path.exists() for path in language_assets),
            },
            'labels': LABELS,
            'label_names': LABEL_NAMES,
            'device': str(self.device),
            'models_loaded': self.models_loaded
        }

    def _extract_audio_features(self, audio_path):
        """Extract signal-level features: MFCC mean/std, pitch stats, energy/rms stats, pauses, repetitions, prolongations"""
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(audio_path, sr=16000, mono=True)

            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1).tolist()
            mfcc_std = np.std(mfcc, axis=1).tolist()

            # Energy (RMS)
            hop_length = 512
            frame_length = 1024
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # Pitch (fundamental frequency) using YIN
            try:
                f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
                voiced_mask = ~np.isnan(f0)
                pitch_mean = float(np.nanmean(f0)) if voiced_mask.any() else 0.0
                pitch_std = float(np.nanstd(f0)) if voiced_mask.any() else 0.0
                voiced_ratio = float(np.sum(voiced_mask) / len(f0))
            except Exception:
                f0 = None
                pitch_mean = 0.0
                pitch_std = 0.0
                voiced_ratio = 0.0

            # Silence / pauses detection
            intervals = librosa.effects.split(y, top_db=30)
            # intervals are non-silent; compute silent gaps
            pauses = []
            if intervals.shape[0] > 0:
                # compute gaps between consecutive non-silent intervals
                for i in range(len(intervals) - 1):
                    gap_samples = intervals[i+1, 0] - intervals[i, 1]
                    gap_seconds = gap_samples / sr
                    if gap_seconds > 0.15:
                        pauses.append(gap_seconds)
            total_pause_time = float(np.sum(pauses)) if pauses else 0.0
            pause_count = int(len(pauses))
            avg_pause = float(np.mean(pauses)) if pauses else 0.0

            # Voiced segments (from voiced_mask if available)
            prolongations = 0
            repetitions = 0
            if 'f0' in locals() and f0 is not None:
                # find contiguous voiced segments
                mask = ~np.isnan(f0)
                if mask.any():
                    # convert frame indices to times
                    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
                    # find contiguous voiced segments
                    segments = []
                    start_idx = None
                    for i, v in enumerate(mask):
                        if v and start_idx is None:
                            start_idx = i
                        if not v and start_idx is not None:
                            segments.append((start_idx, i-1))
                            start_idx = None
                    if start_idx is not None:
                        segments.append((start_idx, len(mask)-1))

                    # analyze segments for prolongations and repetitions
                    for seg in segments:
                        start_t = frame_times[seg[0]]
                        end_t = frame_times[seg[1]] + (frame_length/sr)
                        dur = end_t - start_t
                        if dur >= 1.0:
                            prolongations += 1
                        # short segments separated by tiny gaps may indicate repetitions
                    # simple heuristic for repetitions: count sequences of short voiced segments (<0.35s) separated by gaps <0.15s and length>=2
                    short_segments = []
                    for seg in segments:
                        start_t = frame_times[seg[0]]
                        end_t = frame_times[seg[1]] + (frame_length/sr)
                        dur = end_t - start_t
                        short_segments.append((start_t, end_t, dur))
                    i = 0
                    while i < len(short_segments) - 1:
                        seq_count = 1
                        j = i
                        while j < len(short_segments) - 1:
                            gap = short_segments[j+1][0] - short_segments[j][1]
                            if gap < 0.15 and short_segments[j][2] < 0.35 and short_segments[j+1][2] < 0.35:
                                seq_count += 1
                                j += 1
                            else:
                                break
                        if seq_count >= 2:
                            repetitions += 1
                            i = j + 1
                        else:
                            i += 1

            features = {
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'voiced_ratio': voiced_ratio,
                'pause_count': pause_count,
                'total_pause_time': total_pause_time,
                'avg_pause': avg_pause,
                'prolongations': int(prolongations),
                'repetitions': int(repetitions)
            }

            return features

        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {}
