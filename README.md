**Smart Security Detection System**

This project is a smart person detection system using deep learning.It detects and counts people in an uploaded image.
The system draws bounding boxes around detected persons with confidence scores.
A simple web interface allows users to view results easily.

**Tools & Tecnologies**

1.Python – Programming language

2.PyTorch – Deep learning framework

3.Faster R-CNN – Pretrained object detection model

4.OpenCV – Drawing boxes and text on images

5.Gradio – Web interface for uploading images


**How It Works**

1.The user uploads an image.

2.The model analyzes the image and detects objects.

3.Only people are selected from the detected objects.

4.Bounding boxes are drawn around detected people.

5.The system counts the number of people and shows the result.

6.Person Detection

7.The model is trained on the COCO dataset.

8.It identifies people using a predefined class label.

9.Detections with low confidence are ignored to improve results.


**Confidence Score**

1.Each detected person has a confidence score.

2.The system calculates the average confidence of all detected people.

3.This shows how sure the model is, but it is not actual accuracy.


**User Interface**

A simple Gradio web app allows users to upload images and instantly see detection results.


**Applications**


1.Security and surveillance

2.Crowd monitoring

3.Learning object detection

4.Academic mini projects
