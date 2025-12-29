import streamlit as st
import numpy as np
import joblib
import cv2
from utils.preprocessing import preprocess_image
from utils.features import feature_extraction

# App config
st.set_page_config(page_title="Brain MRI Classifier", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 6px solid #4a90e2;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)
st.title("🧠 Brain MRI Tumor Classifier")
st.markdown("Upload an MRI image and let the model analyze it using texture-based features (GLCM + LBP).")

# model path
bundle = joblib.load("model/model_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# UI Image Uploader
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
process = st.button("Process Image")

# When user press "process" button
if process:
    if uploaded is None:
        st.warning("Please upload an image before pressing Process.")
        st.stop()
    else:
        # Read image
        try:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            st.error("Unable to read the uploaded file as an image.")
            st.stop()
        col1, col2 = st.columns(2)
        with col1:
            # Show input image
            st.subheader("📷 Input Image")
            st.image(image, use_container_width=True, channels="BGR")
        with col2:
            # Preprocess and extract features
            with st.spinner("Preprocessing and extracting features..."):
                img_arr = preprocess_image(image, target_size=(224, 224))
                features = feature_extraction(img_arr)
                features_scaled = scaler.transform(features)
            
            # Predict
            with st.spinner("Running model inference..."):
                pred = int(model.predict(features_scaled)[0])
                label_map = {
                    0: "Pituitary Tumor",
                    1: "No Tumor Detected",
                    2: "Meningioma Tumor",
                    3: "Glioma Tumor"
                }

                class_name = label_map.get(pred, "Unknown")

                # Try to get confidence
                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(features_scaled)[0]
                        confidence = float(np.max(probs))
                    except:
                        pass

            # Display prediction
            st.markdown("### ✅ Prediction Result")
            st.markdown(f"""
            <div class="prediction-card" style="background-color: #3D3D3D;">
                <h3>Predicted Class: <b>{class_name}</b></h3>
                {"<p><b>Confidence:</b> {:.2f}</p>".format(confidence) if confidence else ""}
            </div>
            """, unsafe_allow_html=True)
        
            # Insights panel
            st.markdown("### 📘 What This Result Means")
            if pred == 0:
                st.write("""
                A **Pituitary** tumors develop in the pituitary gland, the body's hormone control center. Most are benign, but they can interfere with hormone production or press on the optic nerve.
                """)
                st.markdown("#### Quick Facts")
                st.write("""
                - Many are non-cancerous (adenomas).
                - They can cause hormone excess or deficiency.
                - They can affect vision because they are located near the optic nerve.      
                """)
                st.markdown("#### Common Symptoms")
                st.write("""
                - Vision problems (especially peripheral vision)
                - Changes in menstrual cycle
                - Extreme fatigue
                - Weight changes
                - Headaches
                """)
                st.markdown("#### General Treatment")
                st.write("""
                - Surgery through the nose (transsphenoidal surgery)
                - Hormone therapy to balance hormones in the body
                - Radiation therapy if the tumor cannot be completely removed    
                """)
            elif pred == 1:
                st.write("""
                This category means that the model does not detect texture patterns resembling tumors in the MRI image.         
                """)
                st.markdown("#### Important Notes")
                st.write("""
                - This is not a medical diagnosis.
                - MRI results must still be evaluated by a radiologist to ensure a comprehensive assessment of the brain's condition.
                """)
            elif pred == 2:
                st.write("""
                A **meningioma** is a tumor that grows from the meninges, which are the protective layers of the brain and spinal cord. Most meningiomas are benign and grow slowly, but they can still put pressure on brain tissue.
                """)
                st.markdown("#### Quick Facts")
                st.write("""
                - It is the most common primary brain tumor in adults.
                - It is more common in women.
                - It usually grows slowly, but it can reach a large size before causing symptoms.      
                """)
                st.markdown("#### Common Symptoms")
                st.write("""
                - Chronic headaches
                - Vision problems (double vision, blurred vision)
                - Hearing loss
                - Muscle weakness
                - Seizures     
                """)
                st.markdown("#### General Treatment")
                st.write("""
                - Surgery (especially if it is pressing on brain tissue)
                - Radiation therapy for tumors that are difficult to remove
                - Regular monitoring for small tumors that do not cause symptoms      
                """)
            else:
                st.write("""
                A **Glioma** is a group of brain tumors that originate from glial cells, which are supporting cells that help neurons function properly. Gliomas can be benign, but most are aggressive and can spread to surrounding brain tissue.
                """)
                st.markdown("#### Quick Facts")
                st.write("""
                - It is one of the most common types of brain tumors.
                - It consists of several subtypes such as astrocytoma, oligodendroglioma, and glioblastoma.
                - The severity varies greatly depending on the grade (I–IV).         
                """)
                st.markdown("#### Common Symptoms")
                st.write("""
                - Increasingly frequent headaches
                - Seizures
                - Changes in behavior or personality
                - Difficulty speaking or understanding language
                - Weakness on one side of the body       
                """)
                st.markdown("#### General Treatment")
                st.write("""
                - Surgery to remove as much tumor tissue as possible
                - Radiotherapy
                - Chemotherapy
                - Targeted therapy (depending on the type of glioma)      
                """)
                
# Recommended next steps (non-medical guidance)
exp1 = st.expander("### 📌 Recommended Next Steps")
exp1.write("""
- This tool is **for research and educational purposes only**.  
- If this MRI is part of a real medical case, please consult a **radiologist or neurologist**.  
- For clinical use, the model must undergo validation on real‑world hospital data.
""")
# Limitations & confidence guidance
exp2 = st.expander("### ⚠️ Limitations")
exp2.write("""
- The model may not generalize to MRI scans from different hospitals or machines.
- Texture‑based models can be sensitive to noise, brightness, and preprocessing differences.  
- Confidence values (if shown) are not medically calibrated.
""")

            