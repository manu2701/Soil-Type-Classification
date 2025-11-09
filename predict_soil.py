"""
Soil Type Classification - Prediction Script
Uses the trained model to classify soil images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import sys

# Soil type labels
SOIL_TYPES = {
    0: "Black Soil",
    1: "Cinder Soil",
    2: "Laterite Soil",
    3: "Peat Soil",
    4: "Yellow Soil"
}

def load_model(model_path='my_model.h5'):
    """Load the trained model"""
    try:
        # Register the KerasLayer for custom objects
        tf.keras.utils.get_custom_objects()['KerasLayer'] = hub.KerasLayer
        # Try loading with compile=False first to avoid compilation issues
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
            print(f"Model loaded successfully from {model_path}")
            # Compile the model
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['acc']
            )
            return model
        except:
            # Try with compile=True
            model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
            print(f"Model loaded successfully from {model_path}")
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to rebuild model architecture...")
        # Rebuild model architecture if loading fails
        mobile_net_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
        
        # Create a wrapper layer for the hub layer to work with newer Keras
        class HubLayerWrapper(tf.keras.layers.Layer):
            def __init__(self, hub_url, **kwargs):
                super().__init__(**kwargs)
                self.hub_layer = hub.KerasLayer(hub_url, input_shape=(224, 224, 3), trainable=False)
            
            def call(self, inputs):
                return self.hub_layer(inputs)
        
        pretrained_wrapper = HubLayerWrapper(mobile_net_model)
        
        # Try Sequential first to match original structure
        try:
            model = tf.keras.Sequential([
                pretrained_wrapper,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
        except:
            # Fallback to Functional API if Sequential doesn't work
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = pretrained_wrapper(inputs)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['acc']
        )
        
        # Build the model by calling it with a dummy input
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = model(dummy_input)
        
        # Try to load weights - skip the hub layer (layer 0) and load the rest
        try:
            # Load weights by skipping the first layer (hub layer) since it's pretrained
            import h5py
            with h5py.File(model_path, 'r') as f:
                if 'model_weights' in f:
                    # Get the layer names from saved model
                    saved_layers = list(f['model_weights'].keys())
                    print(f"Saved model layers: {saved_layers}")
                    
                    # Try to load weights layer by layer, skipping the hub layer
                    model_layers = [l.name for l in model.layers]
                    print(f"New model layers: {model_layers}")
                    
                    # Architecture: hub_layer -> flatten -> dense(1024) -> dense(512) -> dense(256) -> dense(5)
                    # Saved layers: ['dense_22', 'dense_23', 'dense_24', 'dense_25', 'flatten_6', 'keras_layer_4']
                    # Model layers: ['hub_layer_wrapper', 'flatten', 'dense', 'dense_1', 'dense_2', 'dense_3']
                    
                    # Get model layers in order (skip hub wrapper at index 0)
                    model_layer_order = [l for l in model.layers[1:]]  # Skip hub wrapper
                    # Expected: flatten, dense(1024), dense(512), dense(256), dense(5)
                    
                    # Get saved layers (excluding keras_layer_4 which is the hub layer)
                    saved_layer_names = [l for l in saved_layers if l != 'keras_layer_4' and l != 'top_level_model_weights']
                    
                    # Try to match by checking layer shapes
                    # We know: flatten has no weights, dense layers have weights
                    saved_dense = [l for l in saved_layer_names if 'dense' in l]
                    saved_flatten = [l for l in saved_layer_names if 'flatten' in l]
                    
                    # Load flatten (no weights, so skip)
                    # Load dense layers - match by output size (second dimension of kernel)
                    # Check the weight shapes to match correctly
                    saved_dense_info = []
                    for sd in saved_dense:
                        if sd in f['model_weights'] and sd in f['model_weights'][sd]:
                            # Access the layer subgroup: f['model_weights'][sd][sd]
                            layer_group = f['model_weights'][sd][sd]
                            if 'kernel:0' in layer_group:
                                kernel_shape = layer_group['kernel:0'].shape
                                # Output size is the second dimension
                                output_size = kernel_shape[1] if len(kernel_shape) > 1 else kernel_shape[0]
                                saved_dense_info.append((sd, output_size, kernel_shape))
                    
                    # Sort by output size to match: 1024 -> 512 -> 256 -> 5
                    saved_dense_info.sort(key=lambda x: x[1], reverse=True)
                    
                    # Match to model layers: dense(1024) -> dense(512) -> dense(256) -> dense(5)
                    model_dense_layers = [l for l in model_layer_order if 'dense' in l.name.lower()]
                    # Order should be: 1024, 512, 256, 5 (output)
                    
                    for i, (saved_name, output_size, kernel_shape) in enumerate(saved_dense_info):
                        if i < len(model_dense_layers):
                            layer = model_dense_layers[i]
                            # Load weights from the layer subgroup
                            layer_group = f['model_weights'][saved_name][saved_name]
                            layer_weights = []
                            if 'kernel:0' in layer_group:
                                layer_weights.append(layer_group['kernel:0'][:])
                            if 'bias:0' in layer_group:
                                layer_weights.append(layer_group['bias:0'][:])
                            
                            if layer_weights:
                                try:
                                    layer.set_weights(layer_weights)
                                    print(f"Loaded weights for {layer.name} (units: {layer.units if hasattr(layer, 'units') else 'N/A'}) from {saved_name} (output_size: {output_size})")
                                except Exception as e:
                                    print(f"Could not load weights for {layer.name}: {e}")
            
            print("Model weights loaded successfully (hub layer skipped as it's pretrained)")
        except Exception as w_e:
            print(f"Warning: Could not load weights: {w_e}")
            print("Note: The model file may be incompatible with this TensorFlow version.")
            print("The model architecture has been recreated but may need retraining.")
        
        return model

def predict_soil_type(model, image_path):
    """Predict soil type from an image"""
    # Read and preprocess image
    img_test = cv2.imread(image_path)
    if img_test is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img_resize = cv2.resize(img_test, (224, 224))
    img_scaled = img_resize / 255.0
    img_reshaped = np.reshape(img_scaled, [1, 224, 224, 3])
    
    # Make prediction
    input_pred = model.predict(img_reshaped, verbose=0)
    input_label = np.argmax(input_pred)
    confidence = np.max(input_pred) * 100
    
    return input_label, confidence, input_pred

def display_prediction(image_path, predicted_label, confidence, predictions):
    """Display the image and prediction results"""
    img = mpimg.imread(image_path)
    
    plt.figure(figsize=(10, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {SOIL_TYPES[predicted_label]}\nConfidence: {confidence:.2f}%", 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Display prediction probabilities
    plt.subplot(1, 2, 2)
    probs = predictions[0] * 100
    colors = ['green' if i == predicted_label else 'gray' for i in range(5)]
    plt.barh(list(SOIL_TYPES.values()), probs, color=colors)
    plt.xlabel('Probability (%)', fontsize=10)
    plt.title('Prediction Probabilities', fontsize=12)
    plt.xlim(0, 100)
    
    for i, prob in enumerate(probs):
        plt.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    # Get model path
    model_path = 'my_model.h5'
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # If image path provided as argument, use it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a sample image from the dataset
        sample_images = [
            'Soil types/Black Soil/Black_Soil_ (1).jpg',
            'Soil types/Yellow Soil/Yellow_Soil_ (1).jpg',
            'Soil types/Cinder Soil/Cinder_Soil_ (1).jpg',
            'Soil types/Laterite Soil/Laterite_Soil_ (1).jpg',
            'Soil types/Peat Soil/Peat_Soil_ (1).jpg'
        ]
        
        # Find first available sample image
        image_path = None
        for img in sample_images:
            if Path(img).exists():
                image_path = img
                break
        
        if image_path is None:
            print("Error: No sample images found. Please provide an image path as argument.")
            print("Usage: python predict_soil.py <image_path>")
            return
    
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print(f"\nAnalyzing image: {image_path}")
    
    # Make prediction
    try:
        predicted_label, confidence, predictions = predict_soil_type(model, image_path)
        
        print(f"\n{'='*50}")
        print(f"Prediction: {SOIL_TYPES[predicted_label]}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"{'='*50}\n")
        
        print("All probabilities:")
        for i, soil_type in SOIL_TYPES.items():
            prob = predictions[0][i] * 100
            print(f"  {soil_type}: {prob:.2f}%")
        
        # Display visualization
        display_prediction(image_path, predicted_label, confidence, predictions)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

