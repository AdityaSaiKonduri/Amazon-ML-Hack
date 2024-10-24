import numpy as np
from ultralytics import YOLO
import json
import uuid
from PIL import Image, ImageFilter
import os
import easyocr
import shutil

# Initialize EasyOCR Reader (you can specify the language here)
reader = easyocr.Reader(['en'])  # 'en' is for English; you can add other languages if needed

model_path = "runs\\detect\\train\\weights\\best.pt"
image_path = "C:\\Users\\adity\\OneDrive\\Documents\\Amazon ML Hack\\student_resource 3\\images\\51vwYpDz2tL.jpg"

model = YOLO(model_path)
# Run the model on the image
results = model(image_path, conf=0.01, save=True, project=".\\runs\\detect")

# Initialize the output dictionary
output = {"predictions": []}

# Open the original image in grayscale
original_image = Image.open(image_path).convert("L")

# Create a directory to save the extracted images
os.makedirs("extracted_images", exist_ok=True)

# Process each detection
for result in results:
    count = 0
    boxes = result.boxes  # Boxes object for bbox outputs
    for i, box in enumerate(boxes):
        # Get the bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Calculate center x, center y, width, and height
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Get the confidence score
        confidence = box.conf.item()
        
        # Get the class name and id
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        
        # Generate a unique detection ID
        detection_id = str(uuid.uuid4())
        
        # Create a dictionary for this detection
        detection = {
            "x": round(x, 1),
            "y": round(y, 1),
            "width": round(width, 1),
            "height": round(height, 1),
            "confidence": round(confidence, 3),
            "class": class_name,
            "class_id": class_id,
            "detection_id": detection_id
        }
        
        output["predictions"].append(detection)
        
        # Extract the region of interest from the original image
        roi = original_image.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # Sharpen the image
        roi = roi.filter(ImageFilter.SHARPEN)
        
        # Enlarge the image (e.g., 2x)
        new_size = (roi.width * 2, roi.height * 2)
        roi = roi.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save the extracted image
        extracted_image_path = f"extracted_images/{detection_id}_{class_name}.jpg"
        roi.save(extracted_image_path)
        # Add the extracted image path to the detection dictionary
        detection["extracted_image"] = extracted_image_path

# Print the JSON results
print(json.dumps(output, indent=2))

# Save the JSON results to a file
with open("detection_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Results have been saved to detection_results.json")
print("Extracted images have been saved in the 'extracted_images' directory")

# Use EasyOCR to extract text from the saved images
for i in os.listdir("extracted_images"):
    file_name = os.path.join("extracted_images", i)
    img = Image.open(file_name)
    
    # Convert the PIL image to a NumPy array for EasyOCR
    img_np = np.array(img)
    
    # Use EasyOCR to extract text
    result = reader.readtext(img_np, detail=0)  # detail=0 for text-only output
    
    # Combine the text into a single string and print it
    text = ' '.join(result)
    print(f"Output : {text}")

# Delete the extracted images folder
shutil.rmtree("extracted_images")
print("Extracted images folder has been deleted.")
