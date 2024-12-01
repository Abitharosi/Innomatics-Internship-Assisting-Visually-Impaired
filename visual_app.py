
import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import pytesseract
import pyttsx3
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("AIzaSyDfnznFcOxc1fqVaHkOuWa0nMTb_lbbAjk"))

# Streamlit App Configuration
st.set_page_config(page_title="Vision AI", layout="centered", page_icon="🤖")

# Helper functions
def get_response(input_prompt, image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def image_to_bytes(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    except Exception as e:
        raise FileNotFoundError(f"Failed to process image. Please try again. Error: {e}")

def extract_text_from_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(img)
        if not extracted_text.strip():
            return "No text found in the image."
        return extracted_text
    except Exception as e:
        raise ValueError(f"Failed to extract text. Error: {e}")

def text_to_speech_pyttsx3(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        raise RuntimeError(f"Failed to convert text to speech. Error: {e}")

@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.5, iou_threshold=0.5):
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image)
        predictions = object_detection_model([img_tensor])[0]
        keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
        return filtered_predictions
    except Exception as e:
        raise RuntimeError(f"Failed to detect objects. Error: {e}")

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    labels = predictions['labels']
    boxes = predictions['boxes']
    scores = predictions['scores']
    for label, box, score in zip(labels, boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label.item()]
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1), f"{class_name} ({score:.2f})", fill="black")
    return image

# Main Section
st.title("👁️Vision AI App ")
st.write("Welcome to the Vision AI App! This application is designed to assist visually impaired users by analyzing images and providing scene descriptions, object detection, and more.")

# Display Instructions in Right Section
st.markdown("""
### **Features:**
- **🏞️Real-Time Scene Analysis**: Describe scenes from uploaded images.
- **🚧Object and Obstacle Detection**: Detect objects/obstacles for safe navigation.
- **📝Text-to-Speech Conversion**: Convert text to audio descriptions.
- **🤖Personalized Assistance**: Provide task-specific guidance.
""")

# Sidebar with Menu and File Uploader
st.sidebar.header("**MENU**")
# Rearranged menu options as requested
menu_options = ["Describe Scene 🏞️", "Detect Objects 🚧", "Extract Text 📝", "Assist Tasks 🤖"]
selected_action = st.sidebar.selectbox("Choose an action:", menu_options)

st.sidebar.subheader("**UPLOAD IMAGE**")
uploaded_file = st.sidebar.file_uploader("Choose an image:", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Main Functionality
if selected_action == "Describe Scene 🏞️" and uploaded_file:
    with st.spinner("Analyzing Image..."):
        st.subheader("🏞️ Scene Description:")
        image_data = image_to_bytes(uploaded_file)
        response = get_response(
            """
            You are an AI assistant. Analyze the uploaded image and describe its content in simple language.
            """, image_data
        )
        st.write(response)
        text_to_speech_pyttsx3(response)

elif selected_action == "Detect Objects 🚧" and uploaded_file:
    with st.spinner("Detecting objects..."):
        st.subheader("🚧 Detected Objects:")
        image = Image.open(uploaded_file)
        predictions = detect_objects(image)
        image_with_boxes = draw_boxes(image.copy(), predictions)
        st.image(image_with_boxes, caption="Objects Highlighted", use_column_width=True)

elif selected_action == "Extract Text 📝" and uploaded_file:
    with st.spinner("Extracting text from image..."):
        st.subheader("📝 Extracted Text:")
        text = extract_text_from_image(uploaded_file)
        st.write(text)
        if text.strip():
            text_to_speech_pyttsx3(text)

elif selected_action == "Assist Tasks 🤖" and uploaded_file:
    with st.spinner("Providing task-specific assistance..."):
        st.subheader("🤖 Assistance:")
        image_data = image_to_bytes(uploaded_file)
        response = get_response(
            """
            You are a helpful AI assistant. Analyze the uploaded image and suggest tasks for assistance.
            """, image_data
        )
        st.write(response)
        text_to_speech_pyttsx3(response)

# Stop Audio Section
st.sidebar.subheader("**STOP AUDIO**")
stop_audio_button = st.sidebar.button("Stop Audio ⏹️")

if stop_audio_button:
    try:
        if "tts_engine" not in st.session_state:
            st.session_state.tts_engine = pyttsx3.init()
        st.session_state.tts_engine.stop()
        st.success("Audio playback stopped.")
    except Exception as e:
        st.error(f"Failed to stop the audio. Error: {e}")
