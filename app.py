import streamlit as st
import pandas as pd
import numpy as np
from src.visualizations import DataVisualizer
from src.models import ModelLoader
from src.styles import TITLE_STYLE, SIDEBAR_STYLE
from src.streamlit_utils import DataContent, DataTable
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import io
import os
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Fabric Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

def convert_df_to_csv(df):
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

def main():
    st.markdown(TITLE_STYLE, unsafe_allow_html=True)
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

    st.markdown('<h1 class="styled-title">Intelligent Fabric Defect Detection Using Deep Learning and Real-Time Vision Systems Application</h1>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Select Options</div>', unsafe_allow_html=True)
    

    if 'page' not in st.session_state:
        st.session_state['page'] = "Problem Statement"

    if "df" not in st.session_state:
        st.session_state.df = None 
    
    if 'pre_df' not in st.session_state:
        st.session_state.pre_df = None
    
    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None

    # Sidebar buttons
    if st.sidebar.button("Problem Statement"):
        st.session_state['page'] = "Problem Statement"

    if st.sidebar.button("Project Data Description"):
        st.session_state['page'] = "Project Data Description"

    if st.sidebar.button("Sample Training Data"):
        st.session_state['page'] = "Sample Training Data"

    if st.sidebar.button("Data Preprocessing"):
        st.session_state['page'] = "Data Preprocessing"


    if st.sidebar.button("Machine Learning Models Used"):
        st.session_state['page'] = "Machine Learning Models Used"

    if st.sidebar.button("Model Predictions"):
        st.session_state['page'] = "Model Predictions"

################################################################################################################

    if st.session_state['page']== "Problem Statement":
        st.image("./image2.webp", width=700)
        st.markdown(DataContent.problem_statement)
    
    elif  st.session_state['page'] == "Project Data Description":
        st.markdown(DataContent.project_data_details)

    elif st.session_state['page'] == "Sample Training Data":

        st.header("📂 Sample Training Images")

        image_dir = r"C:\fabric_detection\test\images"  # adjust this path

        # List image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:20]  # limit to 12 for demo

        cols = st.columns(4)  # 4 images per row

        for i, image_file in enumerate(image_files):
            img_path = os.path.join(image_dir, image_file)
            with cols[i % 4]:  # cycle through columns
                st.image(Image.open(img_path), caption=image_file, use_container_width=True)


    elif  st.session_state['page'] == "Data Preprocessing":
        st.markdown(DataContent.Data_preprocessing)
        
    
    elif st.session_state['page'] == "Machine Learning Models Used":
        st.markdown(DataContent.ml_models)
        
    
    elif st.session_state['page'] == "Model Predictions":

        MODEL_DIR = r'C:\fabric_detection\runs\detect\train2\weights\best.pt'
        model = YOLO(MODEL_DIR)

        def inference_images(uploaded_file, model):
            image = Image.open(uploaded_file)
            predict = model.predict(image)
            boxes = predict[0].boxes
            plotted = predict[0].plot()[:, :, ::-1]

            # Define defect labels
            defect_labels = {
                0: ("Oil 🛢️", st.warning),
                1: ("Hole 🕳️", st.error),
                2: ("Cutting ✂️", st.info),
                3: ("Crack ⚡", st.success)
            }

            col1, col2 = st.columns([2, 1])  # Image (left), Detection (right)

            with col1:
                st.image(plotted, caption="🖼️ Detected Image", width=500)

            with col2:
                if len(boxes) == 0:
                    st.markdown("**No Detection Found** ❌")
                else:
                    predicted_class = int(predict[0].boxes.cls[0])
                    label, style_func = defect_labels.get(predicted_class, ("Unknown ❓", st.markdown))
                    style_func(f"### 🔍 Detected Defect:\n**{label}**")

        st.write("Upload an image for defect detection")

        uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file and uploaded_file.type.startswith('image'):
            inference_images(uploaded_file, model)


        

if __name__ == "__main__":
    main()