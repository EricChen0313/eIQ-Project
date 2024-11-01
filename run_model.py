import argparse
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time

def load_image(image_path, input_size):
    """Load and preprocess the image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size)
    input_data = np.expand_dims(np.array(image), axis=0)
    input_data = np.float32(input_data)
    return input_data

def run_inference(model_path, image_path, use_npu=False):
    """Run inference on the given image using a TensorFlow Lite model."""
    
    # Load the TensorFlow Lite model
    if use_npu:
        # Load the NPU delegate
        delegate = tflite.load_delegate('libethosu_delegate.so')
        interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    else:
        interpreter = tflite.Interpreter(model_path=model_path)
    
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    input_size = input_details[0]['shape'][1:3]  # Get model input size (height, width)
    input_data = load_image(image_path, input_size)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    start_time = time.time()
    interpreter.invoke()  # Perform the actual inference
    end_time = time.time()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print inference time
    print(f"Inference Time: {end_time - start_time:.3f} seconds")
    
    # Output the model results
    print("Output data:", output_data)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inference on a given image using a TFLite model")
    parser.add_argument('--model', type=str, required=True, help='Path to the TFLite model')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to be classified')
    
    # Change here: Add action='store_true' for boolean flag
    parser.add_argument('--use_npu', action='store_true', help='Enable NPU for inference')
    
    args = parser.parse_args()

    # Run the inference with the provided model and image
    run_inference(args.model, args.image, args.use_npu)