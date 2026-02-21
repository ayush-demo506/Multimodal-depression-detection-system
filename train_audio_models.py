#!/usr/bin/env python3
"""
Audio Depression Detection Model Training Script
Trains multiple models using audio features for depression detection
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Extract comprehensive audio features for depression detection"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Basic audio features
            features = {}
            
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # 3. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 5. Tonnetz Features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            features['tonnetz_std'] = np.std(tonnetz, axis=1)
            
            # 6. Prosody Features (Pitch, Energy, Tempo)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_min'] = 0
                features['pitch_max'] = 0
            
            # Energy features
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            features['energy_max'] = np.max(rms)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 7. Rhythm Features
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_count'] = len(onset_frames)
            features['onset_rate'] = len(onset_frames) / (len(y) / sr)
            
            # 8. Silence Features
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames**2, axis=0)
            silence_threshold = np.percentile(energy, 20)
            silence_frames = np.sum(energy < silence_threshold)
            features['silence_ratio'] = silence_frames / len(energy)
            
            # 9. Voice Quality Features
            # Jitter (pitch period variation)
            if len(pitch_values) > 1:
                jitter = np.std(np.diff(pitch_values)) / np.mean(pitch_values)
                features['jitter'] = jitter
            else:
                features['jitter'] = 0
            
            # Shimmer (amplitude variation)
            frame_energy = np.sum(frames**2, axis=0)
            if len(frame_energy) > 1:
                shimmer = np.std(np.diff(frame_energy)) / np.mean(frame_energy)
                features['shimmer'] = shimmer
            else:
                features['shimmer'] = 0
            
            # 10. Formant-like features
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
            features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

def create_audio_dataset():
    """Create comprehensive audio dataset with features"""
    print("🎵 Creating Audio Dataset...")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # Dataset paths
    base_path = Path("dataset_audio/dataset-depression")
    categories = {
        'depression1': 1,
        'depression2': 1,
        'normal1': 0,
        'normal2': 0
    }
    
    all_features = []
    all_labels = []
    all_files = []
    
    print("📊 Extracting features from audio files...")
    
    for category, label in categories.items():
        category_path = base_path / category
        if not category_path.exists():
            print(f"⚠️  Warning: {category_path} not found, skipping...")
            continue
            
        files = list(category_path.glob("*.wav"))
        print(f"Processing {len(files)} files from {category}...")
        
        for i, file_path in enumerate(files):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(files)} files...")
            
            features = extractor.extract_features(str(file_path))
            if features is not None:
                # Flatten nested arrays
                flat_features = {}
                for key, value in features.items():
                    if isinstance(value, np.ndarray):
                        for j, v in enumerate(value):
                            flat_features[f"{key}_{j}"] = v
                    else:
                        flat_features[key] = value
                
                all_features.append(flat_features)
                all_labels.append(label)
                all_files.append(str(file_path))
    
    print(f"✅ Extracted features from {len(all_features)} audio files")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels
    df['file_path'] = all_files
    
    # Add metadata
    df['category'] = df['file_path'].apply(lambda x: Path(x).parent.name)
    df['filename'] = df['file_path'].apply(lambda x: Path(x).name)
    
    # Handle missing values
    df = df.fillna(0)
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    
    return df

def train_audio_models():
    """Train multiple audio models for depression detection"""
    print("🤖 Training Audio Models...")
    
    # Create dataset
    df = create_audio_dataset()
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['label', 'file_path', 'category', 'filename']]
    X = df[feature_columns]
    y = df['label']
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Number of features: {len(feature_columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"📊 Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'basic_energy': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'description': 'Basic Energy Model - Fast energy and prosody analysis',
            'features': 'Energy, Pitch, Tempo, Basic Spectral'
        },
        'advanced_prosody': {
            'model': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
            'description': 'Advanced Prosody Model - Comprehensive prosodic analysis',
            'features': 'MFCC, Spectral, Prosody, Voice Quality'
        }
    }
    
    results = {}
    
    # Train each model
    for model_name, model_info in models.items():
        print(f"\n🎯 Training {model_name}...")
        
        model = model_info['model']
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Probabilities
        y_proba_train = model.predict_proba(X_train_scaled)
        y_proba_val = model.predict_proba(X_val_scaled)
        y_proba_test = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted')
        }
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_pred_val),
            'precision': precision_score(y_val, y_pred_val, average='weighted'),
            'recall': recall_score(y_val, y_pred_val, average='weighted'),
            'f1': f1_score(y_val, y_pred_val, average='weighted')
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        results[model_name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'description': model_info['description'],
            'features': model_info['features']
        }
        
        print(f"✅ {model_name} - Test Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"   Precision: {test_metrics['precision']:.3f}, Recall: {test_metrics['recall']:.3f}, F1: {test_metrics['f1']:.3f}")
        
        # Save model
        model_dir = Path("models/audio")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Saved model to {model_path}")
    
    # Save scaler
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Saved scaler to {scaler_path}")
    
    # Save results
    results_path = model_dir / "models_report.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved results to {results_path}")
    
    # Save feature names
    feature_names_path = model_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print(f"💾 Saved feature names to {feature_names_path}")
    
    # Print summary
    print("\n📊 MODEL TRAINING SUMMARY")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n🎯 {model_name.upper()}")
        print(f"   Description: {result['description']}")
        print(f"   Features: {result['features']}")
        print(f"   Test Accuracy: {result['test']['accuracy']:.3f}")
        print(f"   Test F1-Score: {result['test']['f1']:.3f}")
    
    return results

def create_audio_suggestions_dataset():
    """Create suggestions dataset for audio analysis"""
    print("🎵 Creating Audio Suggestions Dataset...")
    
    suggestions = [
        {
            "model_type": "basic_energy",
            "analysis_type": "Energy & Prosody",
            "suggestions": {
                "low_energy": [
                    "Consider energy-boosting activities like light exercise or social interaction",
                    "Low energy patterns may indicate depression - consider professional support",
                    "Try engaging in activities that naturally increase energy levels"
                ],
                "high_energy": [
                    "Your energy levels appear healthy - maintain current activities",
                    "Continue engaging in energy-boosting activities",
                    "Monitor for any sudden changes in energy patterns"
                ],
                "pitch_variation": [
                    "Pitch variation is within normal range",
                    "Consider voice exercises to maintain vocal health",
                    "Monitor for any changes in speech patterns"
                ]
            }
        },
        {
            "model_type": "advanced_prosody",
            "analysis_type": "Comprehensive Prosodic Analysis",
            "suggestions": {
                "prosody_healthy": [
                    "Your prosodic features indicate healthy speech patterns",
                    "Continue maintaining good vocal health practices",
                    "Regular speech monitoring is recommended"
                ],
                "prosody_concerning": [
                    "Some prosodic features may indicate stress or depression",
                    "Consider professional speech therapy or mental health support",
                    "Practice relaxation techniques to improve speech patterns"
                ],
                "voice_quality": [
                    "Voice quality analysis shows normal patterns",
                    "Maintain good vocal hygiene and hydration",
                    "Consider voice exercises for optimal vocal health"
                ]
            }
        }
    ]
    
    # Save suggestions
    suggestions_path = Path("models/audio/suggestions_dataset.json")
    with open(suggestions_path, 'w') as f:
        json.dump(suggestions, f, indent=2)
    print(f"💾 Saved suggestions to {suggestions_path}")
    
    return suggestions

if __name__ == "__main__":
    print("🎵 Audio Depression Detection Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path("dataset_audio/dataset-depression").exists():
        print("❌ Audio dataset not found!")
        print("Please ensure the dataset is in: dataset_audio/dataset-depression/")
        exit(1)
    
    # Train models
    results = train_audio_models()
    
    # Create suggestions dataset
    create_audio_suggestions_dataset()
    
    print("\n🎉 Audio Model Training Complete!")
    print("Models saved to: models/audio/")
    print("Ready for integration with Flask app! 🚀")
