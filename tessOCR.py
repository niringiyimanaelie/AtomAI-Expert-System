from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np
import os
from pathlib import Path
import json
import time

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def preprocess_image(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Convert PIL image to a NumPy array for OpenCV operations
    img_np = np.array(img)
    
    # Resize the image by scaling factor of 2
    img_resized = cv2.resize(img_np, (0, 0), fx=2, fy=2)
    
    # Apply Otsu thresholding
    _, img_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL image
    img_pil = Image.fromarray(img_thresh)
    
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(2)
    
    # Apply sharpening filter
    img_pil = img_pil.filter(ImageFilter.SHARPEN)
    
    return img_pil

def extract_text_with_config(image_path, lang='eng'):
    try:
        preprocessed_img = preprocess_image(image_path)
        custom_config = r'--psm 6 -l eng'
        extracted_text = pytesseract.image_to_string(preprocessed_img, lang=lang, config=custom_config)
        return extracted_text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    ocr_results = []
    image_path = r'KE\Images'
    subfolders = [d.strip() for d in os.listdir(image_path)]
    t0 = time.time()
    for s in subfolders:
        if s == 'am_text' or s == 'am_caption' or s == 'am_table':
            cropped_image_path = Path(f"{image_path}/{s}")
            
            for cropped in os.listdir(cropped_image_path):
                extracted_text = extract_text_with_config(str(f"{cropped_image_path}/{cropped}"))
                print(f"Done extracting text from {cropped}.\n")
                ocr_results.append({
                    'doc_number': cropped,
                    'extracted_text': extracted_text
                })
                
    json_dir = r'json_output'
    json_dir = Path(json_dir)
    # Write OCR results to a JSON file
    with open(json_dir / 'ocr_results_ke.json', 'w') as json_file:
        json.dump(ocr_results, json_file, indent=4) 

    print(f'Done. ({time.time() - t0:.3f}s)')