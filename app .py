import gradio as gr
import cv2
import easyocr
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Function to perform text detection and return the image with bounding boxes
def detect_text(image):
    # Convert the image to OpenCV format
    image_cv = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect text on the image
    text_results = reader.readtext(image_cv)

    # Draw bounding boxes and text on the image
    for result in text_results:
        bbox, text, score = result
        if score > 0.25:
            cv2.rectangle(image_cv, bbox[0], bbox[2], (0, 255, 0), 5)
            cv2.putText(image_cv, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    # Convert the processed image back to a format that Gradio can display
    image_with_boxes = cv2.imencode('.jpg', image_cv)[1].tobytes()
    
    return image_with_boxes

# Create a Gradio interface
iface = gr.Interface(
    fn=detect_text,
    inputs=gr.inputs.Image(type="pil", label="Upload an image"),
    outputs=gr.outputs.Image(type="pil"),
    title="EasyOCR Text Detection",
    description="Upload an image and detect text with EasyOCR",
    live=True,
)

# Launch the Gradio app
iface.launch()
