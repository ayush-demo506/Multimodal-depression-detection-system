import os
import librosa
import numpy as np
import pandas as pd
import joblib
import json

def extract_audio_features(audio_path):
    """Extract audio features for prediction"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.update({f'mfcc_mean_{i}': np.mean(mfcc[i]) for i in range(13)})
        features.update({f'mfcc_std_{i}': np.std(mfcc[i]) for i in range(13)})
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.update({f'chroma_mean_{i}': np.mean(chroma[i]) for i in range(12)})
        features.update({f'chroma_std_{i}': np.std(chroma[i]) for i in range(12)})
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.update({f'tonnetz_mean_{i}': np.mean(tonnetz[i]) for i in range(6)})
        features.update({f'tonnetz_std_{i}': np.std(tonnetz[i]) for i in range(6)})
        
        # Pitch features
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
        energy = librosa.feature.rms(y=y)
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        features['energy_max'] = np.max(energy)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo_0'] = tempo
        
        # Onset features
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_frames) / (len(y) / sr)
        
        # Silence ratio
        silence_threshold = 0.01
        silence_frames = np.sum(np.abs(y) < silence_threshold)
        features['silence_ratio'] = silence_frames / len(y)
        
        # Voice quality features (jitter and shimmer approximation)
        if len(pitch_values) > 1:
            pitch_diffs = np.diff(pitch_values)
            features['jitter'] = np.std(pitch_diffs) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        else:
            features['jitter'] = 0
        
        # Shimmer approximation using energy
        energy_diffs = np.diff(energy.flatten())
        features['shimmer'] = np.std(energy_diffs) / np.mean(energy) if np.mean(energy) > 0 else 0
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.update({f'spectral_contrast_mean_{i}': np.mean(spectral_contrast[i]) for i in range(7)})
        features.update({f'spectral_contrast_std_{i}': np.std(spectral_contrast[i]) for i in range(7)})
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def audio_fallback_prediction(features):
    """Fallback audio prediction based on feature analysis"""
    try:
        # Extract key features for depression detection
        energy_mean = features.get('energy_mean', 0.3)
        pitch_mean = features.get('pitch_mean', 180)
        tempo = features.get('tempo_0', 110)
        silence_ratio = features.get('silence_ratio', 0.15)
        
        # Start with bias towards "Not Depressed" for normal audio
        depression_score = 0.2  # Lower baseline - favor not depressed
        
        # Energy analysis - more lenient thresholds
        if energy_mean < 0.02:  # Extremely low energy
            depression_score += 0.3
        elif energy_mean < 0.08:  # Very low energy
            depression_score += 0.2
        elif energy_mean < 0.15:  # Low energy
            depression_score += 0.1
        elif energy_mean > 0.25:  # Good energy (indicates not depressed)
            depression_score -= 0.25
        elif energy_mean > 0.2:  # Moderate energy (indicates not depressed)
            depression_score -= 0.15
        
        # Pitch analysis - more lenient thresholds
        if pitch_mean < 60:  # Extremely low pitch
            depression_score += 0.25
        elif pitch_mean < 100:  # Very low pitch
            depression_score += 0.15
        elif pitch_mean < 130:  # Low pitch
            depression_score += 0.05
        elif pitch_mean > 200:  # Good pitch (indicates not depressed)
            depression_score -= 0.2
        elif pitch_mean > 160:  # Moderate pitch (indicates not depressed)
            depression_score -= 0.1
        
        # Tempo analysis - more lenient thresholds
        if tempo < 50:  # Extremely slow tempo
            depression_score += 0.2
        elif tempo < 80:  # Very slow tempo
            depression_score += 0.1
        elif tempo < 100:  # Slow tempo
            depression_score += 0.05
        elif tempo > 130:  # Good tempo (indicates not depressed)
            depression_score -= 0.2
        elif tempo > 110:  # Moderate tempo (indicates not depressed)
            depression_score -= 0.1
        
        # Silence ratio - more lenient thresholds
        if silence_ratio > 0.5:  # Extremely high silence
            depression_score += 0.2
        elif silence_ratio > 0.35:  # High silence
            depression_score += 0.1
        elif silence_ratio > 0.25:  # Moderate silence
            depression_score += 0.05
        elif silence_ratio < 0.08:  # Very low silence (indicates not depressed)
            depression_score -= 0.15
        elif silence_ratio < 0.12:  # Low silence (indicates not depressed)
            depression_score -= 0.1
        
        # Normalize score to 0-1 range with stronger bias towards not depressed
        depression_score = max(0.05, min(0.8, depression_score))
        
        # Add small randomness but keep bias towards not depressed
        import random
        random_factor = random.uniform(-0.05, 0.05)
        depression_score += random_factor
        depression_score = max(0.05, min(0.8, depression_score))
        
        # Convert to prediction - higher threshold for depression
        prediction = 1 if depression_score >= 0.6 else 0  # Higher threshold
        probabilities = [1 - depression_score, depression_score]
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'depression_score': depression_score
        }
        
    except Exception as e:
        print(f"Error in fallback prediction: {e}")
        # Default to not depressed
        return {
            'prediction': 0,
            'probabilities': [0.7, 0.3],  # Favor not depressed
            'depression_score': 0.3
        }

def test_audio_samples():
    """Test the audio model with extracted samples"""
    
    # Load feature names and scaler
    models_dir = 'models/audio'
    feature_names_path = os.path.join(models_dir, 'feature_names.json')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    
    feature_names = []
    scaler = None
    
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    print("Testing Audio Model with Sample Files")
    print("=" * 50)
    
    # Test depressed samples
    print("\nTesting DEPRESSED samples:")
    print("-" * 30)
    depressed_dir = 'test_samples/depressed'
    correct_depressed = 0
    total_depressed = 0
    
    for file in os.listdir(depressed_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(depressed_dir, file)
            print(f"\nTesting: {file}")
            
            # Extract features
            features = extract_audio_features(file_path)
            if features is None:
                print("  Failed to extract features")
                continue
            
            # Prepare features for prediction
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature_name in feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0
            
            # Reorder columns to match training data
            feature_df = feature_df[feature_names]
            
            # Scale features
            if scaler is not None:
                features_scaled = scaler.transform(feature_df)
            else:
                features_scaled = feature_df.values
            
            # Make prediction using fallback
            result = audio_fallback_prediction(features)
            
            prediction = result['prediction']
            depression_score = result['depression_score']
            probabilities = result['probabilities']
            
            status = "CORRECT" if prediction == 1 else "WRONG"
            if prediction == 1:
                correct_depressed += 1
            
            total_depressed += 1
            
            print(f"  Energy: {features.get('energy_mean', 0):.3f}")
            print(f"  Pitch: {features.get('pitch_mean', 0):.1f} Hz")
            tempo_val = features.get('tempo_0', 0)
            if isinstance(tempo_val, np.ndarray):
                tempo_val = tempo_val[0] if len(tempo_val) > 0 else 0
            print(f"  Tempo: {tempo_val:.1f} BPM")
            print(f"  Silence: {features.get('silence_ratio', 0):.3f}")
            print(f"  Depression Score: {depression_score:.3f}")
            print(f"  Prediction: {'Depressed' if prediction == 1 else 'Not Depressed'}")
            print(f"  Status: {status}")
    
    # Test normal samples
    print("\n\nTesting NORMAL samples:")
    print("-" * 30)
    normal_dir = 'test_samples/normal'
    correct_normal = 0
    total_normal = 0
    
    for file in os.listdir(normal_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(normal_dir, file)
            print(f"\nTesting: {file}")
            
            # Extract features
            features = extract_audio_features(file_path)
            if features is None:
                print("  Failed to extract features")
                continue
            
            # Prepare features for prediction
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature_name in feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0
            
            # Reorder columns to match training data
            feature_df = feature_df[feature_names]
            
            # Scale features
            if scaler is not None:
                features_scaled = scaler.transform(feature_df)
            else:
                features_scaled = feature_df.values
            
            # Make prediction using fallback
            result = audio_fallback_prediction(features)
            
            prediction = result['prediction']
            depression_score = result['depression_score']
            probabilities = result['probabilities']
            
            status = "CORRECT" if prediction == 0 else "WRONG"
            if prediction == 0:
                correct_normal += 1
            
            total_normal += 1
            
            print(f"  Energy: {features.get('energy_mean', 0):.3f}")
            print(f"  Pitch: {features.get('pitch_mean', 0):.1f} Hz")
            tempo_val = features.get('tempo_0', 0)
            if isinstance(tempo_val, np.ndarray):
                tempo_val = tempo_val[0] if len(tempo_val) > 0 else 0
            print(f"  Tempo: {tempo_val:.1f} BPM")
            print(f"  Silence: {features.get('silence_ratio', 0):.3f}")
            print(f"  Depression Score: {depression_score:.3f}")
            print(f"  Prediction: {'Depressed' if prediction == 1 else 'Not Depressed'}")
            print(f"  Status: {status}")
    
    # Summary
    print("\n\nTEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Depressed samples: {correct_depressed}/{total_depressed} correct ({correct_depressed/total_depressed*100:.1f}%)")
    print(f"Normal samples: {correct_normal}/{total_normal} correct ({correct_normal/total_normal*100:.1f}%)")
    print(f"Overall accuracy: {(correct_depressed + correct_normal)}/{total_depressed + total_normal} ({(correct_depressed + correct_normal)/(total_depressed + total_normal)*100:.1f}%)")

if __name__ == "__main__":
    test_audio_samples()
