import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import gradio as gr

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

transform = transforms.ToTensor()
CONFIDENCE_THRESHOLD = 0.6

def detect_people(image):
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(img)[0]

    person_count = 0
    confidences = []

    for i in range(len(output["scores"])):
        if output["labels"][i] == 1 and output["scores"][i] > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = output["boxes"][i].int().tolist()
            score = output["scores"][i].item()
            confidences.append(score)
            person_count += 1

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"Person {score:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    accuracy = np.mean(confidences)*100 if confidences else 0

    cv2.putText(frame, f"People: {person_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (20,75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, f"People: {person_count} | Accuracy: {accuracy:.1f}%"

demo = gr.Interface(
    fn=detect_people,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(label="Detection Output"),
        gr.Textbox(label="Result")
    ],
    title="Smart Security Person Detection",
    description="Upload an image to detect people using Faster R-CNN"
)

demo.launch()