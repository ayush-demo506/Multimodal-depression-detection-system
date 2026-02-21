import os
import shutil
import random

def extract_test_samples():
    """Extract sample audio files for testing the model"""
    
    # Source directories
    depressed_dirs = [
        'dataset_audio/dataset-depression/depression1',
        'dataset_audio/dataset-depression/depression2'
    ]
    
    normal_dirs = [
        'dataset_audio/dataset-depression/normal1',
        'dataset_audio/dataset-depression/normal2'
    ]
    
    # Create test samples directory
    test_dir = 'test_samples'
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories
    depressed_test_dir = os.path.join(test_dir, 'depressed')
    normal_test_dir = os.path.join(test_dir, 'normal')
    os.makedirs(depressed_test_dir, exist_ok=True)
    os.makedirs(normal_test_dir, exist_ok=True)
    
    # Extract depressed samples
    print("Extracting depressed audio samples...")
    depressed_count = 0
    for dir_path in depressed_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            # Take first 5 files from each directory
            sample_files = files[:5]
            for file in sample_files:
                src = os.path.join(dir_path, file)
                dst = os.path.join(depressed_test_dir, f"depressed_{depressed_count + 1}_{file}")
                shutil.copy2(src, dst)
                depressed_count += 1
                print(f"  Copied: {file} -> depressed_{depressed_count}_{file}")
    
    # Extract normal samples
    print("\nExtracting normal audio samples...")
    normal_count = 0
    for dir_path in normal_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            # Take first 5 files from each directory
            sample_files = files[:5]
            for file in sample_files:
                src = os.path.join(dir_path, file)
                dst = os.path.join(normal_test_dir, f"normal_{normal_count + 1}_{file}")
                shutil.copy2(src, dst)
                normal_count += 1
                print(f"  Copied: {file} -> normal_{normal_count}_{file}")
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 Depressed samples: {depressed_count} files")
    print(f"📁 Normal samples: {normal_count} files")
    print(f"📁 Total test samples: {depressed_count + normal_count} files")
    print(f"📂 Test samples saved in: {test_dir}/")
    
    # Create a summary file
    summary_file = os.path.join(test_dir, 'test_samples_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Audio Test Samples Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Depressed samples: {depressed_count}\n")
        f.write(f"Normal samples: {normal_count}\n")
        f.write(f"Total samples: {depressed_count + normal_count}\n\n")
        
        f.write("Depressed files:\n")
        for i in range(1, depressed_count + 1):
            f.write(f"  depressed_{i}_*.wav\n")
        
        f.write("\nNormal files:\n")
        for i in range(1, normal_count + 1):
            f.write(f"  normal_{i}_*.wav\n")
    
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    extract_test_samples()
