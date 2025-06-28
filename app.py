

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np


class LightSoilNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LightSoilNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]

    r, g, b = cv2.split(image)
    r = cv2.addWeighted(r, 0.7, v_channel, 0.3, 0)
    g = cv2.addWeighted(g, 0.7, v_channel, 0.3, 0)
    b = cv2.addWeighted(b, 0.7, v_channel, 0.3, 0)

    enhanced = cv2.merge([r, g, b])
    enhanced = cv2.resize(enhanced, (224, 224))
    image = Image.fromarray(enhanced)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


@st.cache_resource
def load_model():
    model = LightSoilNet(num_classes=4)
    model.load_state_dict(torch.load("lightsoilnet_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


st.title("🌱 Soil Type Classification")
st.write("Upload a soil image and classify it into one of the following:")
class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
st.write("**Types**:", ", ".join(class_names))

uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    model = load_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        threshold = 0.7  # Confidence threshold

        if confidence.item() < threshold:
            st.error(" No Soil Detected - Please upload a valid soil image.")
        else:
            predicted_class = class_names[predicted.item()]
            st.success(f" Predicted Soil Type: **{predicted_class}**")
            st.write(f" Confidence: **{confidence.item():.2f}**")
