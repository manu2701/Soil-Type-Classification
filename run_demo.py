"""
Quick demo script to test the model on multiple sample images
"""

import os
from pathlib import Path
from predict_soil import load_model, predict_soil_type, SOIL_TYPES

def main():
    """Run predictions on sample images from each soil type"""
    print("="*60)
    print("Soil Type Classification - Demo")
    print("="*60)
    
    # Load model
    model_path = 'my_model.h5'
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    print("\nLoading model...")
    model = load_model(model_path)
    
    # Test images from each category
    test_images = {
        "Black Soil": "Soil types/Black Soil/Black_Soil_ (1).jpg",
        "Yellow Soil": "Soil types/Yellow Soil/Yellow_Soil_ (1).jpg",
        "Cinder Soil": "Soil types/Cinder Soil/Cinder_Soil_ (1).jpg",
        "Laterite Soil": "Soil types/Laterite Soil/Laterite_Soil_ (1).jpg",
        "Peat Soil": "Soil types/Peat Soil/Peat_Soil_ (1).jpg"
    }
    
    print("\n" + "="*60)
    print("Running predictions on sample images...")
    print("="*60 + "\n")
    
    correct = 0
    total = 0
    
    for expected_type, image_path in test_images.items():
        if not Path(image_path).exists():
            print(f"⚠️  Skipping {expected_type}: Image not found")
            continue
        
        try:
            predicted_label, confidence, predictions = predict_soil_type(model, image_path)
            predicted_type = SOIL_TYPES[predicted_label]
            
            is_correct = predicted_type == expected_type
            status = "✅" if is_correct else "❌"
            
            print(f"{status} {expected_type}")
            print(f"   Predicted: {predicted_type} ({confidence:.2f}%)")
            if not is_correct:
                print(f"   Expected: {expected_type}")
            print()
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"❌ Error processing {expected_type}: {e}\n")
    
    print("="*60)
    print(f"Results: {correct}/{total} correct predictions")
    if total > 0:
        print(f"Accuracy: {(correct/total)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()

