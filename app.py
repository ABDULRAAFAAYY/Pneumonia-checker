import os
# Force PyTorch only - avoid TensorFlow conflicts
os.environ["USE_TF"] = "0"

import streamlit as st
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0f0f2f 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #00d4ff;
        font-size: 3rem;
        font-weight: 800;
    }
    
    .sub-header {
        text-align: center;
        color: #a0a0c0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .upload-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Result cards */
    .result-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 95, 70, 0.3));
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        animation: pulse-green 2s infinite;
    }
    
    .result-pneumonia {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(127, 29, 29, 0.3));
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.6); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.6); }
    }
    
    /* Confidence bar */
    .confidence-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the model (cached for performance)
@st.cache_resource
def load_model():
    """Load the pneumonia detection model from Hugging Face."""
    try:
        from transformers import pipeline
        # Using a working ViT model for chest X-ray classification
        classifier = pipeline(
            "image-classification",
            model="nickmuchi/vit-finetuned-chest-xray-pneumonia"
        )
        return classifier, None
    except Exception as e:
        return None, str(e)

# Main UI
st.markdown('<h1 class="main-header">🫁 Pneumonia Detection AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image for AI-powered pneumonia detection</p>', unsafe_allow_html=True)

# Info section
st.markdown("""
<div class="info-box">
    <strong>📋 How it works:</strong><br>
    1. Upload a chest X-ray image (PNG, JPG, JPEG)<br>
    2. Our AI model analyzes the image<br>
    3. Get instant results with confidence score
</div>
""", unsafe_allow_html=True)

# Load model with status
with st.spinner("🔄 Loading AI model (first time may take a minute)..."):
    classifier, error = load_model()

if error:
    st.error(f"Failed to load model: {error}")
    st.info("Please ensure you have installed the requirements: `pip install -r requirements.txt`")
else:
    st.success("✅ Model loaded successfully!")

# File uploader
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a clear chest X-ray image for best results"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and classifier is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Uploaded X-Ray")
        image = Image.open(uploaded_file)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### 🔬 Analysis Result")
        
        with st.spinner("🔄 Analyzing image with AI..."):
            try:
                # Run prediction
                results = classifier(image)
                
                if results:
                    # Get top prediction
                    top_result = results[0]
                    label = top_result['label']
                    confidence = top_result['score'] * 100
                    
                    # Determine if it's pneumonia or normal
                    is_pneumonia = 'pneumonia' in label.lower()
                    
                    if is_pneumonia:
                        st.markdown(f"""
                        <div class="result-pneumonia">
                            <h2 style="color: #ef4444; margin: 0;">⚠️ PNEUMONIA DETECTED</h2>
                            <p style="color: #fca5a5; margin-top: 0.5rem;">Signs of pneumonia found in the X-ray</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal">
                            <h2 style="color: #10b981; margin: 0;">✅ NORMAL</h2>
                            <p style="color: #6ee7b7; margin-top: 0.5rem;">No signs of pneumonia detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence score
                    st.markdown(f"""
                    <div class="confidence-container">
                        <strong>Confidence Score:</strong> {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence / 100)
                    
                    # Show all predictions
                    if len(results) > 1:
                        with st.expander("📊 Detailed Results"):
                            for pred in results:
                                st.write(f"**{pred['label']}**: {pred['score']*100:.2f}%")
                else:
                    st.warning("Could not get prediction results.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong><br>
    This AI tool is for educational and research purposes only. It should NOT be used as a substitute 
    for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #6b7280;">
    Powered by Hugging Face 🤗 Transformers | Built with Streamlit ❤️
</p>
""", unsafe_allow_html=True)

