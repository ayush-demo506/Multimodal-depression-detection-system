#!/usr/bin/env python3
"""
Comprehensive Dataset Generator for Multimodal Depression Detection.
Creates large, detailed datasets with suggestions and analysis capabilities from scratch.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
import json
import re

# Sample depressed and positive texts for dataset generation
DEPRESSED_TEXTS = [
    "I feel so empty inside, nothing brings me joy anymore",
    "Every day feels like a struggle, I can't find motivation",
    "I'm tired all the time, even simple tasks feel overwhelming",
    "I don't want to see anyone, I just want to be alone",
    "Nothing seems to matter anymore, what's the point",
    "I can't sleep properly, my mind won't stop racing",
    "I feel like I'm a burden to everyone around me",
    "I've lost interest in everything I used to love",
    "I feel hopeless about the future",
    "I can't concentrate on anything, my mind feels foggy",
    "I feel worthless and inadequate",
    "I'm constantly worried and anxious about everything",
    "I feel disconnected from everyone around me",
    "I don't have the energy to do anything",
    "I feel like I'm drowning in sadness",
    "I can't stop crying, everything makes me emotional",
    "I feel like I'm trapped in a dark place",
    "I don't see any way out of this situation",
    "I feel like I'm losing myself",
    "I can't remember the last time I felt happy",
    "I feel like I'm broken beyond repair",
    "I don't want to get out of bed anymore",
    "I feel like I'm a failure at everything",
    "I can't shake this feeling of despair",
    "I feel like I'm invisible to the world",
    "I don't have any hope left",
    "I feel like I'm suffocating",
    "I can't find any meaning in life",
    "I feel like I'm slowly disappearing",
    "I don't know how to be happy anymore"
]

POSITIVE_TEXTS = [
    "I'm feeling grateful for all the good things in my life",
    "Today was a great day, I accomplished so much",
    "I love spending time with my friends and family",
    "I'm excited about the new opportunities coming my way",
    "I feel energized and ready to take on challenges",
    "I'm proud of the progress I've made recently",
    "I enjoy my hobbies and they bring me joy",
    "I feel confident and optimistic about the future",
    "I love learning new things and growing as a person",
    "I feel connected to the people around me",
    "I'm happy with who I am becoming",
    "I find joy in the simple pleasures of life",
    "I feel motivated to pursue my goals",
    "I appreciate the beauty in everyday moments",
    "I feel peaceful and content with my life",
    "I enjoy helping others and making a difference",
    "I feel strong and capable of handling anything",
    "I'm excited about new adventures and experiences",
    "I feel blessed to have such wonderful people in my life",
    "I love the feeling of accomplishment after hard work",
    "I feel inspired and creative",
    "I enjoy exploring new places and cultures",
    "I feel proud of my achievements",
    "I love the feeling of being productive",
    "I feel grateful for my health and well-being",
    "I enjoy the company of positive people",
    "I feel hopeful about what the future holds",
    "I love the feeling of being understood and supported",
    "I feel confident in my abilities",
    "I enjoy making others smile and laugh"
]

def generate_detailed_user_context(is_depressed, severity_level):
    """Generate comprehensive user context based on depression status and severity"""
    
    # Age distribution (more realistic)
    age = random.randint(18, 65)
    
    # Gender distribution
    gender = random.choice(["Male", "Female", "Other", ""])
    
    # Platform preferences
    platforms = ["Twitter/X", "Instagram", "Facebook", "Reddit", "WhatsApp", "Other"]
    platform = random.choice(platforms)
    
    # Timeframe
    timeframes = ["Last 24 hours", "Last 7 days", "Last 30 days"]
    timeframe = random.choice(timeframes)
    
    # Detailed well-being scales based on depression severity
    if is_depressed:
        if severity_level == "severe":
            sleep_quality = random.choices(["Good", "Average", "Poor"], weights=[0.05, 0.15, 0.8])[0]
            energy = random.randint(1, 2)  # Very low energy
            mood = random.randint(1, 3)     # Very low mood
            anxiety = random.randint(8, 10) # Very high anxiety
            stress = random.randint(8, 10) # Very high stress
            
            symptoms = random.sample([
                "Persistent sadness", "Loss of interest", "Anxiety", "Irritability",
                "Hopelessness", "Sleep issues", "Fatigue", "Appetite change", "Poor concentration"
            ], k=random.randint(6, 9))
            
            events = random.choice([
                "Major life crisis", "Loss of loved one", "Severe health issues", "Job loss",
                "Relationship breakdown", "Financial crisis", "Traumatic event", "Social isolation"
            ])
            
        elif severity_level == "moderate":
            sleep_quality = random.choices(["Good", "Average", "Poor"], weights=[0.1, 0.4, 0.5])[0]
            energy = random.randint(2, 3)  # Low energy
            mood = random.randint(2, 4)     # Low mood
            anxiety = random.randint(6, 8) # High anxiety
            stress = random.randint(6, 8) # High stress
            
            symptoms = random.sample([
                "Persistent sadness", "Loss of interest", "Anxiety", "Irritability",
                "Hopelessness", "Sleep issues", "Fatigue", "Appetite change", "Poor concentration"
            ], k=random.randint(4, 6))
            
            events = random.choice([
                "Work stress", "Relationship issues", "Health concerns", "Academic pressure",
                "Family problems", "Social difficulties", "Career uncertainty", "Personal setbacks"
            ])
            
        else:  # mild
            sleep_quality = random.choices(["Good", "Average", "Poor"], weights=[0.2, 0.5, 0.3])[0]
            energy = random.randint(3, 4)  # Moderate energy
            mood = random.randint(3, 5)     # Moderate mood
            anxiety = random.randint(4, 6) # Moderate anxiety
            stress = random.randint(4, 6) # Moderate stress
            
            symptoms = random.sample([
                "Persistent sadness", "Loss of interest", "Anxiety", "Irritability",
                "Hopelessness", "Sleep issues", "Fatigue", "Appetite change", "Poor concentration"
            ], k=random.randint(2, 4))
            
            events = random.choice([
                "Minor setbacks", "Daily stress", "Social challenges", "Work pressure",
                "Personal concerns", "Health worries", "Relationship tension", "Life transitions"
            ])
    else:
        sleep_quality = random.choices(["Good", "Average", "Poor"], weights=[0.6, 0.3, 0.1])[0]
        energy = random.randint(4, 5)  # High energy
        mood = random.randint(7, 10)   # High mood
        anxiety = random.randint(1, 4)  # Low anxiety
        stress = random.randint(1, 4)  # Low stress
        
        symptoms = random.sample([
            "Persistent sadness", "Loss of interest", "Anxiety", "Irritability",
            "Hopelessness", "Sleep issues", "Fatigue", "Appetite change", "Poor concentration"
        ], k=random.randint(0, 1))
        
        events = random.choice([
            "New opportunities", "Positive changes", "Career success", "Personal growth",
            "Healthy relationships", "New hobbies", "Travel plans", "Achievements", ""
        ])
    
    return {
        "age": age,
        "gender": gender,
        "platform": platform,
        "timeframe": timeframe,
        "sleep": sleep_quality,
        "energy": energy,
        "mood": mood,
        "anxiety": anxiety,
        "stress": stress,
        "symptoms": symptoms,
        "events": events
    }

def generate_detailed_suggestions(is_depressed, severity_level, context):
    """Generate detailed suggestions based on analysis"""
    
    suggestions = []
    analysis_points = []
    
    if is_depressed:
        if severity_level == "severe":
            suggestions.extend([
                "Consider immediate professional help - therapist or counselor",
                "Reach out to trusted friends or family members",
                "Practice deep breathing exercises daily",
                "Maintain a regular sleep schedule",
                "Consider medication consultation with psychiatrist",
                "Join support groups for depression",
                "Engage in light physical activity when possible",
                "Limit social media consumption",
                "Practice mindfulness meditation",
                "Consider crisis hotline if thoughts become overwhelming"
            ])
            analysis_points.extend([
                "High risk indicators detected",
                "Multiple symptoms present",
                "Significant impact on daily functioning",
                "Professional intervention recommended",
                "Support system activation needed"
            ])
            
        elif severity_level == "moderate":
            suggestions.extend([
                "Schedule appointment with mental health professional",
                "Practice regular self-care routines",
                "Engage in social activities with trusted people",
                "Maintain consistent sleep patterns",
                "Consider therapy or counseling",
                "Practice stress management techniques",
                "Engage in enjoyable activities",
                "Limit alcohol and substance use",
                "Practice gratitude journaling",
                "Consider support groups"
            ])
            analysis_points.extend([
                "Moderate risk level identified",
                "Several symptoms present",
                "Some impact on daily life",
                "Professional support beneficial",
                "Self-care strategies important"
            ])
            
        else:  # mild
            suggestions.extend([
                "Practice daily mindfulness or meditation",
                "Maintain regular exercise routine",
                "Ensure adequate sleep hygiene",
                "Engage in social connections",
                "Practice stress reduction techniques",
                "Consider therapy for prevention",
                "Monitor mood patterns",
                "Engage in hobbies and interests",
                "Practice positive self-talk",
                "Maintain healthy lifestyle habits"
            ])
            analysis_points.extend([
                "Mild symptoms detected",
                "Early intervention beneficial",
                "Preventive measures recommended",
                "Lifestyle adjustments helpful",
                "Monitor for changes"
            ])
    else:
        suggestions.extend([
            "Continue current positive practices",
            "Maintain healthy lifestyle habits",
            "Stay connected with supportive people",
            "Engage in regular physical activity",
            "Practice stress management techniques",
            "Maintain work-life balance",
            "Continue engaging in enjoyable activities",
            "Practice gratitude and mindfulness",
            "Support others who may be struggling",
            "Regular mental health check-ins"
        ])
        analysis_points.extend([
            "Positive mental health indicators",
            "Good coping strategies in place",
            "Healthy lifestyle maintained",
            "Strong support system present",
            "Continue current practices"
        ])
    
    # Add context-specific suggestions
    if context["sleep"] == "Poor":
        suggestions.append("Focus on sleep hygiene - consistent bedtime, no screens before bed")
    if context["energy"] <= 2:
        suggestions.append("Consider gentle physical activity to boost energy levels")
    if context["anxiety"] >= 7:
        suggestions.append("Practice anxiety management techniques - breathing exercises, grounding")
    if context["stress"] >= 7:
        suggestions.append("Implement stress reduction strategies - time management, relaxation")
    
    return {
        "suggestions": suggestions[:8],  # Top 8 suggestions
        "analysis_points": analysis_points[:5],  # Top 5 analysis points
        "risk_level": severity_level if is_depressed else "low",
        "recommended_action": "professional_help" if severity_level == "severe" else "self_care" if not is_depressed else "monitoring"
    }

def create_comprehensive_samples(num_samples=15000):
    """Create comprehensive samples with detailed analysis"""
    
    samples = []
    
    # Expand text pools
    depressed_texts = DEPRESSED_TEXTS * 50  # 1500 texts
    positive_texts = POSITIVE_TEXTS * 50    # 1500 texts
    
    random.shuffle(depressed_texts)
    random.shuffle(positive_texts)
    
    for i in range(num_samples):
        # Determine depression status and severity
        is_depressed = random.choice([True, False])
        
        if is_depressed:
            severity_level = random.choices(["mild", "moderate", "severe"], weights=[0.4, 0.4, 0.2])[0]
        else:
            severity_level = "low"
        
        # Generate comprehensive context
        context = generate_detailed_user_context(is_depressed, severity_level)
        
        # Generate detailed suggestions and analysis
        suggestions_data = generate_detailed_suggestions(is_depressed, severity_level, context)
        
        # Select texts (2-10 per sample)
        num_texts = random.randint(2, 10)
        
        # Filter texts based on depression status
        if is_depressed:
            available_texts = depressed_texts
        else:
            available_texts = positive_texts
        
        # Sample texts for this user
        selected_texts = random.sample(available_texts, min(num_texts, len(available_texts)))
        
        # Pad with empty strings if needed
        while len(selected_texts) < 10:
            selected_texts.append("")
        
        # Create comprehensive sample
        sample = {
            "user_info": {
                "name": f"User_{i+1:05d}",
                "age": context["age"],
                "gender": context["gender"],
                "contact": f"user{i+1}@example.com"
            },
            "context": {
                "platform": context["platform"],
                "timeframe": context["timeframe"],
                "events": context["events"]
            },
            "wellbeing": {
                "sleep": context["sleep"],
                "energy": context["energy"],
                "mood": context["mood"],
                "anxiety": context["anxiety"],
                "stress": context["stress"],
                "symptoms": context["symptoms"]
            },
            "texts": {
                f"text_{j+1}": selected_texts[j] for j in range(10)
            },
            "analysis": {
                "risk_level": suggestions_data["risk_level"],
                "severity": severity_level,
                "analysis_points": suggestions_data["analysis_points"],
                "recommended_action": suggestions_data["recommended_action"]
            },
            "suggestions": suggestions_data["suggestions"],
            "label": 1 if is_depressed else 0,
            "label_text": "Depressed" if is_depressed else "Not Depressed",
            "confidence_score": random.uniform(0.7, 0.95) if is_depressed else random.uniform(0.8, 0.98)
        }
        
        samples.append(sample)
    
    return samples

def save_comprehensive_datasets(samples):
    """Save comprehensive datasets in multiple formats"""
    
    output_dir = Path("data_for_text")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame for CSV export
    df_data = []
    for sample in samples:
        row = {
            "user_name": sample["user_info"]["name"],
            "age": sample["user_info"]["age"],
            "gender": sample["user_info"]["gender"],
            "contact": sample["user_info"]["contact"],
            "platform": sample["context"]["platform"],
            "timeframe": sample["context"]["timeframe"],
            "events": sample["context"]["events"],
            "sleep": sample["wellbeing"]["sleep"],
            "energy": sample["wellbeing"]["energy"],
            "mood": sample["wellbeing"]["mood"],
            "anxiety": sample["wellbeing"]["anxiety"],
            "stress": sample["wellbeing"]["stress"],
            "symptoms": ",".join(sample["wellbeing"]["symptoms"]),
            "risk_level": sample["analysis"]["risk_level"],
            "severity": sample["analysis"]["severity"],
            "analysis_points": "|".join(sample["analysis"]["analysis_points"]),
            "recommended_action": sample["analysis"]["recommended_action"],
            "suggestions": "|".join(sample["suggestions"]),
            "label": sample["label"],
            "label_text": sample["label_text"],
            "confidence_score": sample["confidence_score"]
        }
        
        # Add text fields
        for i in range(1, 11):
            row[f"text_{i}"] = sample["texts"][f"text_{i}"]
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save comprehensive CSV
    csv_path = output_dir / "comprehensive_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved comprehensive CSV dataset: {csv_path}")
    
    # Save JSON (for easy loading in training)
    json_path = output_dir / "comprehensive_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved comprehensive JSON dataset: {json_path}")
    
    # Save training/validation/test split (70/20/10)
    train_size = int(0.7 * len(df))
    val_size = int(0.2 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    train_df.to_csv(output_dir / "train_comprehensive.csv", index=False)
    val_df.to_csv(output_dir / "val_comprehensive.csv", index=False)
    test_df.to_csv(output_dir / "test_comprehensive.csv", index=False)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save suggestions dataset separately
    suggestions_data = []
    for sample in samples:
        for suggestion in sample["suggestions"]:
            suggestions_data.append({
                "user_id": sample["user_info"]["name"],
                "risk_level": sample["analysis"]["risk_level"],
                "severity": sample["analysis"]["severity"],
                "suggestion": suggestion,
                "category": "general" if "professional" not in suggestion.lower() else "professional"
            })
    
    suggestions_df = pd.DataFrame(suggestions_data)
    suggestions_df.to_csv(output_dir / "suggestions_dataset.csv", index=False)
    
    # Print comprehensive statistics
    print(f"\n=== COMPREHENSIVE DATASET STATISTICS ===")
    print(f"Total samples: {len(df)}")
    print(f"Depressed: {len(df[df['label'] == 1])}")
    print(f"Not Depressed: {len(df[df['label'] == 0])}")
    print(f"Average texts per sample: {df[[f'text_{i}' for i in range(1, 11)]].apply(lambda x: (x != '').sum(), axis=1).mean():.1f}")
    
    print(f"\n=== RISK LEVEL DISTRIBUTION ===")
    print(df['risk_level'].value_counts())
    
    print(f"\n=== SEVERITY DISTRIBUTION ===")
    print(df['severity'].value_counts())
    
    print(f"\n=== RECOMMENDED ACTIONS ===")
    print(df['recommended_action'].value_counts())
    
    print(f"\n=== SUGGESTIONS DATASET ===")
    print(f"Total suggestions: {len(suggestions_df)}")
    print(f"Professional suggestions: {len(suggestions_df[suggestions_df['category'] == 'professional'])}")
    print(f"General suggestions: {len(suggestions_df[suggestions_df['category'] == 'general'])}")

def main():
    """Main function"""
    print("Creating comprehensive multimodal depression detection dataset...")
    
    # Create comprehensive samples
    print(f"\nGenerating comprehensive samples...")
    samples = create_comprehensive_samples(num_samples=15000)
    
    # Save datasets
    print(f"\nSaving comprehensive datasets...")
    save_comprehensive_datasets(samples)
    
    print("\n=== DATASET CREATION COMPLETE! ===")
    print("\nFiles created:")
    print("- comprehensive_dataset.csv (full dataset with analysis)")
    print("- comprehensive_dataset.json (structured format)")
    print("- train_comprehensive.csv (70% for training)")
    print("- val_comprehensive.csv (20% for validation)")
    print("- test_comprehensive.csv (10% for testing)")
    print("- suggestions_dataset.csv (detailed suggestions)")
    
    print("\n=== FEATURES INCLUDED ===")
    print("✓ User demographics and context")
    print("✓ Comprehensive well-being assessment")
    print("✓ Multiple text inputs per user")
    print("✓ Risk level analysis")
    print("✓ Severity assessment")
    print("✓ Detailed suggestions")
    print("✓ Recommended actions")
    print("✓ Confidence scores")
    print("✓ Analysis points")

if __name__ == "__main__":
    main()
