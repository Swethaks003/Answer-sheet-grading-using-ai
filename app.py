from flask import Flask, request, render_template, jsonify
from flask import send_from_directory
import os
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import re
import difflib
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize pipelines with error handling
try:
    print("Initializing OCR pipeline...")
    ocr = pipeline('image-to-text', model="microsoft/trocr-base-handwritten", framework="pt")
    print("OCR pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing OCR pipeline: {e}")
    ocr = None

try:
    print("Initializing QA pipeline...")
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
    print("QA pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing QA pipeline: {e}")
    qa_pipeline = None

# Helper functions
def calculate_clarity_score(ocr_results):
    clarity_scores = []
    for result in ocr_results:
        text = result["text"]
        confidence = result.get("confidence", 1.0)
        if len(text.strip()) > 0:
            clarity_scores.append(confidence)
    clarity_score = round(np.mean(clarity_scores) * 10, 2) if clarity_scores else 0
    return clarity_score

def calculate_neatness_score(binary_img):
    white_pixel_ratio = np.sum(binary_img == 255) / binary_img.size
    neatness_score = max(0, min(10, 10 - (white_pixel_ratio - 0.5) * 20))
    return round(neatness_score, 2)

def clean_text_for_comparison(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def calculate_accuracy_score(correct_answer, handwritten_answer):
    correct_answer = clean_text_for_comparison(correct_answer)
    handwritten_answer = clean_text_for_comparison(handwritten_answer)
    sequence_matcher = difflib.SequenceMatcher(None, correct_answer, handwritten_answer)
    accuracy_score = round(sequence_matcher.ratio() * 10, 2)
    return accuracy_score

def preprocess_image(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Apply Gaussian blur to reduce noise
    denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    binary_img = cv2.adaptiveThreshold(
        denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return binary_img

def segment_lines(image, min_line_height=10, min_gap=15):
    horizontal_proj = np.sum(image, axis=1)
    threshold = np.max(horizontal_proj) * 0.2
    line_indices = np.where(horizontal_proj > threshold)[0]
    line_boxes = []
    y_start = None
    for i in range(len(line_indices)):
        if y_start is None:
            y_start = line_indices[i]
        if i == len(line_indices) - 1 or line_indices[i + 1] > line_indices[i] + min_gap:
            y_end = line_indices[i]
            if y_end - y_start >= min_line_height:
                line_boxes.append((0, y_start, image.shape[1], y_end))
            y_start = None
    return line_boxes

def segment_words(line_img):
    vertical_proj = np.sum(line_img, axis=0)
    threshold = np.max(vertical_proj) * 0.2
    word_indices = np.where(vertical_proj > threshold)[0]
    word_boxes = []
    x_start = None
    for i in range(len(word_indices)):
        if x_start is None:
            x_start = word_indices[i]
        if i == len(word_indices) - 1 or word_indices[i + 1] > word_indices[i] + 5:
            x_end = word_indices[i]
            word_boxes.append((x_start, 0, x_end, line_img.shape[0]))
            x_start = None
    return word_boxes

def recognize_lines_and_words(image, line_boxes, visualize=False):
    ocr_results = []
    segmented_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for idx, (x, y_start, width, y_end) in enumerate(line_boxes):
        try:
            line_img = image[y_start:y_end, x:x + width]
            word_boxes = segment_words(line_img)
            for wx, wy, ww, wh in word_boxes:
                word_img = line_img[wy:wy + wh, wx:wx + ww]
                word_pil = Image.fromarray(word_img)
                recognized_text = ocr(word_pil)[0]['generated_text']
                ocr_results.append({"line_number": idx + 1, "bounding_box": (wx, wy + y_start, ww, wh), "text": recognized_text})
                if visualize:
                    cv2.rectangle(segmented_img, (wx, y_start), (wx + ww, y_start + wh), (0, 255, 0), 2)
        except Exception as e:
            ocr_results.append({"line_number": idx + 1, "bounding_box": (x, y_start, width, y_end - y_start), "text": f"OCR Failed: {e}"})

    if visualize:
        plt.imshow(segmented_img)
        plt.axis('off')
        plt.show()

    return ocr_results

def clean_ocr_text(text):
    # Remove all numbers (if not needed)
    text = re.sub(r'\d+', '', text)
    
    # Remove unnecessary sentences
    sentences = text.split('.')
    filtered_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0 and not sentence.strip().isdigit()]
    cleaned_text = '. '.join(filtered_sentences)
    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# Updated grading function to prioritize accuracy and reduce grade if accuracy is low
def grade_answer(correct_answer, handwritten_answer, ocr_results, binary_img):
    clarity_score = calculate_clarity_score(ocr_results)
    neatness_score = calculate_neatness_score(binary_img)
    accuracy_score = calculate_accuracy_score(correct_answer, handwritten_answer)
    
    # Assign weights to each score, prioritizing accuracy
    clarity_weight = 0.2
    neatness_weight = 0.2
    accuracy_weight = 0.6
    
    # Define the accuracy threshold
    accuracy_threshold = 5.0
    
    # Calculate the final grade based on the weighted scores
    final_grade = (clarity_score * clarity_weight) + (neatness_score * neatness_weight) + (accuracy_score * accuracy_weight)
    
    # If accuracy is below the threshold, reduce the final grade and set a flag
    if accuracy_score < accuracy_threshold:
        final_grade = 2.0  # Significantly reduce the grade
        grade_status = "Not Graded"
    else:
        grade_status = "Graded"
    
    clarity_feedback = "The handwriting is exceptionally clear and easy to read. Great effort in presentation!"
    if clarity_score < 7:
        clarity_feedback = "The handwriting could be clearer. Consider writing more legibly to improve readability."

    neatness_feedback = "The handwriting is very well-structured with consistent spacing and alignment."
    if neatness_score < 7:
        neatness_feedback = "The handwriting appears untidy. Improving spacing and alignment can enhance neatness."

    accuracy_feedback = "The handwritten answer does not align with the correct answer. Focus on relevance and comprehension."
    if accuracy_score > 7:
        accuracy_feedback = "The handwritten answer is highly accurate and aligns well with the correct answer. Excellent work!"

    final_grade_feedback = "Great clarity and neatness! Focus more on accuracy to improve your overall score."
    if final_grade > 8:
        final_grade_feedback = "Excellent work! Your answer demonstrates clarity, neatness, and accuracy."
    elif final_grade < 5:
        final_grade_feedback = "Significant improvement is needed in accuracy and neatness to raise your overall grade."

    return {
        "clarity_score": clarity_score,
        "neatness_score": neatness_score,
        "accuracy_score": accuracy_score,
        "final_grade": round(final_grade, 2),
        "grade_status": grade_status,
        "feedback": {
            "clarity": clarity_feedback,
            "neatness": neatness_feedback,
            "accuracy": accuracy_feedback,
            "final_grade": final_grade_feedback
        }
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    question = request.form.get('question')
    if not question:
        return "No question provided", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if os.path.exists(file_path):
        print(f"File saved successfully at {file_path}")
    else:
        print("File not saved!")
        return "File not saved", 500

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read the image!")
        return "Failed to read the image", 500

    print(f"Image dimensions: {img.shape}")

    binary_img = preprocess_image(img)
    line_boxes = segment_lines(binary_img)
    print(f"Line boxes: {line_boxes}")  # Debug statement to check line_boxes
    if not line_boxes:
        print("No lines detected!")
        return "No lines detected in the image", 500

    ocr_results = recognize_lines_and_words(img, line_boxes)
    if not ocr_results:
        print("OCR failed to recognize any text!")
        return "OCR failed to recognize any text", 500
    else:
        print(f"OCR Results: {ocr_results}")

    handwritten_answer = " ".join([line["text"] for line in ocr_results])
    
    # Clean the recognized text
    cleaned_handwritten_answer = clean_ocr_text(handwritten_answer)
    
    concise_question = f"Please provide a concise explanation of {question} in three to five sentences."

    if qa_pipeline:
        correct_answer = qa_pipeline(concise_question, max_length=500, num_return_sequences=1, repetition_penalty=2.0)[0]["generated_text"]
        print(f"Correct Answer: {correct_answer}")
    else:
        print("QA pipeline not initialized!")
        return "QA pipeline not initialized", 500

    evaluation = grade_answer(correct_answer, cleaned_handwritten_answer, ocr_results, binary_img)
    print(f"Evaluation: {evaluation}")

    results_text = f"""
    Question: {question}
    Correct Answer: {correct_answer}
    Handwritten Answer: {handwritten_answer}
    Clarity Score: {evaluation['clarity_score']}/10
    Neatness Score: {evaluation['neatness_score']}/10
    Accuracy Score: {evaluation['accuracy_score']}/10
    Final Grade: {evaluation['final_grade']}/10
    Feedback:
    - Clarity: {evaluation['feedback']['clarity']}
    - Neatness: {evaluation['feedback']['neatness']}
    - Accuracy: {evaluation['feedback']['accuracy']}
    - Final Grade: {evaluation['feedback']['final_grade']}
    """

    return render_template(
        'result.html',
        question=question,
        correct_answer=correct_answer,
        handwritten_answer=cleaned_handwritten_answer,
        clarity_score=evaluation['clarity_score'],
        neatness_score=evaluation['neatness_score'],
        accuracy_score=evaluation['accuracy_score'],
        final_grade=evaluation['final_grade'],
        grade_status=evaluation['grade_status'],
        clarity_feedback=evaluation['feedback']['clarity'],
        neatness_feedback=evaluation['feedback']['neatness'],
        accuracy_feedback=evaluation['feedback']['accuracy'],
        final_grade_feedback=evaluation['feedback']['final_grade']
    )
@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__== "__main__":
    app.run(debug=True,host="127.0.0.1", port=5000)