# MedScript: Intelligent Prescription Digitization System

## Overview

MedScript is an end-to-end system that converts handwritten and printed medical prescriptions into structured digital data using OCR and intelligent post-processing techniques.

The system addresses a real-world problem in healthcare where prescriptions are often unstructured, difficult to read, and manually processed, leading to inefficiencies and errors.

---

## Why This Matters

* Manual prescription entry is time-consuming and error-prone
* Handwritten prescriptions are often unclear or inconsistent
* Lack of structured data makes analysis and tracking difficult

MedScript automates this workflow by extracting and structuring prescription data, improving accuracy and efficiency.

---

## Key Features

* Extracts text from prescription images using OCR (Tesseract + EasyOCR)
* Identifies medicines using dictionary matching and fuzzy logic
* Extracts dosage, frequency, and duration automatically
* Assigns confidence scores to each detected medicine
* Handles noisy OCR outputs using fallback and correction mechanisms
* Works with both printed and handwritten prescriptions

---

## System Workflow

1. Upload prescription image
2. Preprocess image (grayscale, noise reduction)
3. Extract text using OCR engines
4. Identify medicines using dictionary + fuzzy matching
5. Extract dosage, frequency, and duration
6. Assign confidence scores
7. Return structured output

---

## Tech Stack

* Python
* EasyOCR
* Tesseract OCR
* OpenCV
* NumPy / Pandas

---

## Project Structure

```
MedScript/
├── app.py
├── models.py
├── ocr_pipeline.py
├── medication_dict.py
├── requirements.txt
├── templates/
├── uploads/
└── README.md
```

---

## Example Output

```python
{
    'medicine': 'paracetamol',
    'confidence': 0.85,
    'dosage': '500 mg',
    'frequency': 'twice daily',
    'duration': '5 days'
}
```

(Add screenshots here)

---

## Challenges & Learnings

* Handwritten text caused OCR inaccuracies → improved using fallback logic and fuzzy matching
* Medicine names vary widely → handled using dictionary with aliases
* Noisy image inputs → improved using preprocessing techniques
* Structuring unstructured text required custom parsing logic

---

## Limitations

* Accuracy depends on image quality
* Handwritten prescriptions may still produce errors
* Requires Tesseract installation

---

## Future Improvements

* Improve handwriting recognition accuracy
* Add support for multi-language prescriptions
* Integrate with database for patient records
* Build a web dashboard for real-time usage

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
python app.py
```

3. Upload a prescription image and view results

---

## Author

Developed as part of a real-world problem-solving project focused on healthcare data digitization.
