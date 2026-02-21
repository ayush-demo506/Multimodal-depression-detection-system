#!/usr/bin/env python3
"""
Audio Dataset Preparation Script
Creates a comprehensive dataset from audio files for depression detection
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_audio_metadata():
    """Create metadata for audio files"""
    print("🎵 Creating Audio Dataset Metadata...")
    
    base_path = Path("dataset_audio/dataset-depression")
    
    # Define categories and labels
    categories = {
        'depression1': {'label': 1, 'severity': 'moderate', 'description': 'Moderate depression symptoms'},
        'depression2': {'label': 1, 'severity': 'severe', 'description': 'Severe depression symptoms'},
        'normal1': {'label': 0, 'severity': 'healthy', 'description': 'Healthy control group 1'},
        'normal2': {'label': 0, 'severity': 'healthy', 'description': 'Healthy control group 2'}
    }
    
    all_files = []
    
    for category, info in categories.items():
        category_path = base_path / category
        if not category_path.exists():
            print(f"⚠️  Warning: {category_path} not found, skipping...")
            continue
            
        files = list(category_path.glob("*.wav"))
        print(f"Found {len(files)} files in {category}")
        
        for file_path in files:
            try:
                # Get basic audio info
                y, sr = librosa.load(str(file_path), sr=None)
                duration = len(y) / sr
                
                file_info = {
                    'file_path': str(file_path),
                    'filename': file_path.name,
                    'category': category,
                    'label': info['label'],
                    'severity': info['severity'],
                    'description': info['description'],
                    'duration': duration,
                    'sample_rate': sr,
                    'samples': len(y)
                }
                
                all_files.append(file_info)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(all_files)
    
    print(f"📊 Total files processed: {len(df)}")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    print(f"📊 Category distribution:")
    print(df['category'].value_counts())
    
    # Save metadata
    metadata_path = Path("models/audio/audio_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)
    
    print(f"💾 Saved metadata to {metadata_path}")
    
    return df

def create_audio_features_summary():
    """Create a summary of audio features that will be extracted"""
    print("🎵 Creating Audio Features Summary...")
    
    features_summary = {
        "mfcc_features": {
            "description": "Mel-frequency cepstral coefficients - capture spectral envelope",
            "count": 13,
            "importance": "High - captures vocal tract characteristics"
        },
        "spectral_features": {
            "description": "Spectral centroid, rolloff, bandwidth - capture spectral shape",
            "count": 6,
            "importance": "High - indicates voice quality and timbre"
        },
        "prosody_features": {
            "description": "Pitch, energy, tempo - capture speech rhythm and intonation",
            "count": 8,
            "importance": "Very High - directly related to emotional state"
        },
        "voice_quality_features": {
            "description": "Jitter, shimmer, silence ratio - capture voice stability",
            "count": 4,
            "importance": "High - indicates stress and emotional state"
        },
        "rhythm_features": {
            "description": "Onset detection, rhythm patterns - capture speech timing",
            "count": 3,
            "importance": "Medium - indicates cognitive load and emotional state"
        },
        "chroma_features": {
            "description": "Chroma and tonnetz - capture harmonic content",
            "count": 18,
            "importance": "Medium - indicates emotional expression"
        }
    }
    
    # Save features summary
    features_path = Path("models/audio/audio_features_summary.json")
    with open(features_path, 'w') as f:
        json.dump(features_summary, f, indent=2)
    
    print(f"💾 Saved features summary to {features_path}")
    
    return features_summary

def create_audio_analysis_guide():
    """Create a guide for audio analysis interpretation"""
    print("🎵 Creating Audio Analysis Guide...")
    
    analysis_guide = {
        "basic_energy_model": {
            "description": "Fast energy and prosody analysis for quick screening",
            "features_used": ["energy", "pitch", "tempo", "basic_spectral"],
            "interpretation": {
                "low_energy": "May indicate depression, fatigue, or low mood",
                "high_energy": "Generally positive, indicates good mental state",
                "pitch_variation": "Reduced variation may indicate flat affect",
                "tempo_slow": "Slower speech may indicate depression or cognitive load"
            },
            "recommendations": {
                "low_energy": "Consider energy-boosting activities, professional support",
                "high_energy": "Maintain current activities, monitor for changes",
                "pitch_flat": "Consider voice exercises, emotional expression activities"
            }
        },
        "advanced_prosody_model": {
            "description": "Comprehensive prosodic analysis with detailed insights",
            "features_used": ["mfcc", "spectral", "prosody", "voice_quality", "rhythm"],
            "interpretation": {
                "prosody_healthy": "Normal speech patterns, good emotional expression",
                "prosody_concerning": "May indicate stress, depression, or cognitive issues",
                "voice_quality_poor": "May indicate physical or mental health issues",
                "rhythm_irregular": "May indicate cognitive load or emotional distress"
            },
            "recommendations": {
                "prosody_healthy": "Continue current practices, regular monitoring",
                "prosody_concerning": "Consider professional support, stress management",
                "voice_quality_poor": "Consider medical evaluation, voice therapy"
            }
        }
    }
    
    # Save analysis guide
    guide_path = Path("models/audio/audio_analysis_guide.json")
    with open(guide_path, 'w') as f:
        json.dump(analysis_guide, f, indent=2)
    
    print(f"💾 Saved analysis guide to {guide_path}")
    
    return analysis_guide

def create_audio_suggestions():
    """Create comprehensive suggestions for audio analysis"""
    print("🎵 Creating Audio Suggestions...")
    
    suggestions = {
        "immediate_actions": {
            "low_energy_detected": [
                "Consider engaging in light physical activity",
                "Try social interaction or calling a friend",
                "Consider professional mental health support"
            ],
            "high_energy_detected": [
                "Maintain current healthy activities",
                "Continue monitoring for any changes",
                "Consider sharing positive energy with others"
            ],
            "prosody_concerning": [
                "Practice relaxation techniques",
                "Consider voice exercises or singing",
                "Monitor stress levels and seek support if needed"
            ]
        },
        "lifestyle_tips": {
            "vocal_health": [
                "Stay hydrated for better voice quality",
                "Practice regular voice exercises",
                "Avoid excessive vocal strain"
            ],
            "emotional_expression": [
                "Engage in activities that allow emotional expression",
                "Consider music, art, or creative activities",
                "Practice mindfulness and emotional awareness"
            ],
            "social_connection": [
                "Maintain regular social interactions",
                "Practice active listening and communication",
                "Consider joining groups or communities"
            ]
        },
        "professional_resources": {
            "speech_therapy": [
                "Consider speech therapy for voice quality issues",
                "Voice exercises can improve prosodic features",
                "Professional assessment for persistent issues"
            ],
            "mental_health": [
                "Consider counseling or therapy for emotional support",
                "Professional evaluation for depression symptoms",
                "Support groups for shared experiences"
            ],
            "medical_evaluation": [
                "Consider medical evaluation for voice changes",
                "Rule out physical causes of voice issues",
                "Regular health check-ups recommended"
            ]
        },
        "self_care_activities": {
            "vocal_exercises": [
                "Practice humming and vocal warm-ups",
                "Try singing or chanting exercises",
                "Breathing exercises for better voice control"
            ],
            "emotional_wellness": [
                "Practice gratitude and positive thinking",
                "Engage in activities that bring joy",
                "Maintain regular sleep and nutrition"
            ],
            "communication_skills": [
                "Practice active listening",
                "Engage in meaningful conversations",
                "Consider public speaking or presentation skills"
            ]
        }
    }
    
    # Save suggestions
    suggestions_path = Path("models/audio/audio_suggestions.json")
    with open(suggestions_path, 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    print(f"💾 Saved suggestions to {suggestions_path}")
    
    return suggestions

if __name__ == "__main__":
    print("🎵 Audio Dataset Preparation")
    print("=" * 50)
    
    # Check if dataset exists
    if not Path("dataset_audio/dataset-depression").exists():
        print("❌ Audio dataset not found!")
        print("Please ensure the dataset is in: dataset_audio/dataset-depression/")
        exit(1)
    
    # Create metadata
    metadata_df = create_audio_metadata()
    
    # Create features summary
    features_summary = create_audio_features_summary()
    
    # Create analysis guide
    analysis_guide = create_audio_analysis_guide()
    
    # Create suggestions
    suggestions = create_audio_suggestions()
    
    print("\n🎉 Audio Dataset Preparation Complete!")
    print("📁 Files created:")
    print("  - models/audio/audio_metadata.json")
    print("  - models/audio/audio_features_summary.json")
    print("  - models/audio/audio_analysis_guide.json")
    print("  - models/audio/audio_suggestions.json")
    print("\n🚀 Ready to train audio models!")
