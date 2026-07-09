import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Alzheimer MRI Detection",
    layout="centered"
)

# --------------------------------
# Title
# --------------------------------
st.title("🧠 Alzheimer MRI Detection System")

st.write(
    "Upload an MRI Brain Scan Image to Predict Alzheimer Stage"
)

# --------------------------------
# Load Model
# --------------------------------
try:
    model = load_model("alzheimer_model.keras")

    st.success("✅ Model Loaded Successfully")

except Exception as e:

    st.error(f"❌ Error Loading Model:\n{e}")

    st.stop()

# --------------------------------
# Upload Image
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file)

    # Convert RGB
    image = image.convert("RGB")

    # Display Image
    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Convert to Array
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (128, 128))

    # Normalize
    img = img / 255.0

    # Reshape
    img = np.reshape(img, [1, 128, 128, 3])

    # Predict
    prediction = model.predict(img)

    # Classes
    classes = [
        "Mild Dementia",
        "Moderate Dementia",
        "Non Demented",
        "Very Mild Dementia"
    ]

    # Result
    result = classes[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # --------------------------------
    # Output
    # --------------------------------
    st.subheader("Prediction Result")

    st.success(f"🧠 Result: {result}")

    st.info(f"📊 Confidence: {confidence:.2f}%")

    # --------------------------------
    # Probability Scores
    # --------------------------------
    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(classes):

        st.write(
            f"{class_name}: {prediction[0][i] * 100:.2f}%"
        )

# --------------------------------
# Footer
# --------------------------------
st.write("---")

st.write("Developed using CNN, TensorFlow and Streamlit")
