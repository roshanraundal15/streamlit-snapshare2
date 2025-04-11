import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# ------------------ CONFIG -------------------
DATASET_PATH = "dataset"
QUALITY_LABELS = ["good", "average", "bad"]
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# ------------------ ORB Feature Extractor -------------------
orb = cv2.ORB_create(nfeatures=500)

def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if desc1 is None or desc2 is None:
        return 0
    matches = bf.match(desc1, desc2)
    return len(matches)

# ------------------ Prediction -------------------
def predict_quality(uploaded_img):
    uploaded_img = cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2BGR)
    uploaded_desc = extract_orb_features(uploaded_img)

    best_score = -1
    predicted_quality = "Unknown"

    for quality in QUALITY_LABELS:
        quality_dir = os.path.join(DATASET_PATH, quality)
        for filename in os.listdir(quality_dir):
            if filename.lower().endswith(IMG_EXTENSIONS):
                ref_img_path = os.path.join(quality_dir, filename)
                ref_img = cv2.imread(ref_img_path)
                ref_img = cv2.resize(ref_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_desc = extract_orb_features(ref_img)
                match_score = match_descriptors(uploaded_desc, ref_desc)

                if match_score > best_score:
                    best_score = match_score
                    predicted_quality = quality

    return predicted_quality, best_score

# ------------------ Fallback Explanation -------------------
def generate_explanation(quality_label):
    if quality_label == "good":
        return "This crop is of good quality. It is likely to fetch a higher price in the market and indicates healthy farming practices."
    elif quality_label == "average":
        return "This crop is of average quality. It may be sold at a standard price, but improvements in farming or storage could help."
    elif quality_label == "bad":
        return "This crop is of poor quality. It could reduce the market price and may indicate issues like pest damage, improper irrigation, or storage problems."
    else:
        return "Unable to provide explanation due to unknown quality."

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Crop Quality Predictor", page_icon="🌾")
st.title("Quality Check")
st.write("Upload an image of a crop and get predicted quality using ORB feature matching.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    if st.button("Predict Quality"):
        label, score = predict_quality(image)
        st.success(f"🧠 Predicted Quality: **{label.upper()}** (Similarity Score: {score})")

        # Explanation
        explanation = generate_explanation(label)
        st.info(f"📘 Explanation: {explanation}")
