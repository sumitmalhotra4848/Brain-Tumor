"""
Brain MRI Tumor Classifier
Deep Learning based classification of brain tumors with Grad-CAM visualization
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .confidence-high {
        color: #00ff88;
        font-size: 2rem;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-label {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .sample-btn {
        background: #f0f2f6;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .sample-btn:hover {
        background: #e0e2e6;
        transform: translateY(-2px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        transition: 0.3s;
    }
    .probability-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model Architecture
# ============================================================

class BrainTumorClassifier(nn.Module):
    """CNN classifier for brain tumor detection and classification"""
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Get intermediate features for Grad-CAM"""
        # Store intermediate activations
        self.activations = []
        self.gradients = []
        
        # Block 1
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        self.activations.append(x)
        x = self.features[6](x)
        
        # Block 2
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        self.activations.append(x)
        x = self.features[13](x)
        
        # Block 3
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)
        x = self.features[19](x)
        self.activations.append(x)
        x = self.features[20](x)
        
        # Block 4 (last conv layer for Grad-CAM)
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)
        x = self.features[24](x)
        x = self.features[25](x)
        self.last_conv = x
        x = self.features[26](x)
        x = self.features[27](x)
        
        features = x.view(x.size(0), -1)
        output = self.classifier(features)
        return output


# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradient)
        
    def __call__(self, x, class_idx=None):
        # Register hook
        handle = self.target_layer.register_forward_hook(self.forward_hook)
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        handle.remove()
        
        return cam.detach().cpu().numpy()[0, 0], class_idx


# ============================================================
# Utility Functions
# ============================================================

class_names = ["No Tumor", "Pituitary", "Glioma", "Meningioma"]
class_descriptions = {
    "No Tumor": "Healthy brain tissue. No tumor detected in the scan.",
    "Pituitary": "Pituitary adenoma - typically benign tumor of the pituitary gland.",
    "Glioma": "Glioma - tumor arising from glial cells. May require further evaluation.",
    "Meningioma": "Meningioma - usually benign tumor arising from the meninges."
}

class_colors = {
    "No Tumor": "#00ff88",
    "Pituitary": "#f39c12",
    "Glioma": "#e74c3c",
    "Meningioma": "#3498db"
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = BrainTumorClassifier(num_classes=4)
    
    # For demo purposes, create random weights
    # In production, load pretrained weights:
    # model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, (str, bytes)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to tensor and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Normalize using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, image


def predict(model, image_tensor, use_gradcam=True):
    """Run prediction with optional Grad-CAM"""
    model.eval()
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    # Generate Grad-CAM
    cam_image = None
    if use_gradcam:
        # Get last convolutional layer
        target_layer = model.features[25]  # Last conv block
        gradcam = GradCAM(model, target_layer)
        cam, _ = gradcam(image_tensor, pred_class)
        cam_image = cam
    
    return {
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy(),
        'logits': logits[0].cpu().numpy(),
        'gradcam': cam_image
    }


def create_gradcam_overlay(image, cam, alpha=0.5):
    """Create Grad-CAM overlay on original image"""
    # Resize cam to image size
    cam_resized = np.array(Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR))
    
    # Normalize cam to [0, 1]
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_norm)[:, :, :3]
    
    # Convert image to array
    img_array = np.array(image) / 255.0
    
    # Overlay
    overlay = (1 - alpha) * img_array + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_probability_chart(probabilities):
    """Create probability bar chart using plotly"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=[class_colors[c] for c in class_names],
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='outside',
            name='Probability'
        )
    ])
    
    fig.update_layout(
        title="Per-class Probability",
        xaxis_title="Tumor Type",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def generate_pdf_report(image, prediction, confidence, probabilities, gradcam_overlay):
    """Generate PDF report of the analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    # Build document
    story = []
    
    # Title
    story.append(Paragraph("Brain MRI Tumor Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Timestamp
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Prediction results
    story.append(Paragraph("Diagnosis Summary", heading_style))
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", normal_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.1f}%", normal_style))
    story.append(Paragraph(f"<b>Clinical Note:</b> {class_descriptions[prediction]}", normal_style))
    story.append(Spacer(1, 20))
    
    # Save and add images
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        image.save(tmp_img.name)
        img = ReportImage(tmp_img.name, width=3*inch, height=3*inch)
        story.append(Paragraph("Input MRI Scan", heading_style))
        story.append(img)
        story.append(Spacer(1, 10))
        os.unlink(tmp_img.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_grad:
        gradcam_img = Image.fromarray((gradcam_overlay * 255).astype(np.uint8))
        gradcam_img.save(tmp_grad.name)
        grad_img = ReportImage(tmp_grad.name, width=3*inch, height=3*inch)
        story.append(Paragraph("Grad-CAM Heatmap", heading_style))
        story.append(grad_img)
        story.append(Spacer(1, 10))
        os.unlink(tmp_grad.name)
    
    # Probabilities table
    story.append(Paragraph("Per-class Probabilities", heading_style))
    prob_data = [["Tumor Type", "Probability"]]
    for name, prob in zip(class_names, probabilities):
        prob_data.append([name, f"{prob*100:.1f}%"])
    
    prob_table = Table(prob_data, colWidths=[2.5*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", heading_style))
    story.append(Paragraph(
        "This analysis is for research purposes only and should not be used for clinical diagnosis. "
        "Always consult with a qualified medical professional for medical decisions.",
        normal_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# Main App UI
# ============================================================

def main():
    # Header
    st.markdown('<h1 class="main-title">🧠 Brain MRI Tumor Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a brain MRI scan and get a tumor-type prediction explained with Grad-CAM.</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Create two columns for layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        st.markdown("## Choose an MRI image")
        
        # Upload section
        st.markdown("**Upload your scan**")
        uploaded_file = st.file_uploader(
            "Upload",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        st.caption("10MB per file · JPG, JPEG, PNG")
        
        # Sample images section
        st.markdown("**Or try a sample**")
        sample_cols = st.columns(3)
        
        sample_images = {
            "Sample 1": "https://raw.githubusercontent.com/sartajbhuvaji/Brain-Tumor-Classification-Using-Deep-Learning/main/Data/Testing/pituitary/Te-pi_0010.jpg",
            "Sample 2": "https://raw.githubusercontent.com/sartajbhuvaji/Brain-Tumor-Classification-Using-Deep-Learning/main/Data/Testing/glioma/Te-gl_0053.jpg",
            "Sample 3": "https://raw.githubusercontent.com/sartajbhuvaji/Brain-Tumor-Classification-Using-Deep-Learning/main/Data/Testing/meningioma/Te-me_0019.jpg"
        }
        
        for idx, (name, url) in enumerate(sample_images.items()):
            with sample_cols[idx]:
                if st.button(name, key=f"sample_{idx}", use_container_width=True):
                    try:
                        import requests
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            image = Image.open(io.BytesIO(response.content))
                            st.session_state.current_image = image
                            
                            # Process prediction
                            model = load_model()
                            img_tensor, processed_img = preprocess_image(image)
                            result = predict(model, img_tensor, use_gradcam=True)
                            st.session_state.prediction_result = result
                            st.session_state.current_processed_img = processed_img
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample: {e}")
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                
                # Process prediction
                with st.spinner("Analyzing MRI scan..."):
                    model = load_model()
                    img_tensor, processed_img = preprocess_image(image)
                    result = predict(model, img_tensor, use_gradcam=True)
                    st.session_state.prediction_result = result
                    st.session_state.current_processed_img = processed_img
                st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with right_col:
        st.markdown("## Result")
        
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            
            # Prediction display
            prediction = result['prediction']
            confidence = result['confidence']
            color = class_colors[prediction]
            
            st.markdown(f"""
            <div class="result-card">
                <p style="color: #888; margin-bottom: 0;">Prediction</p>
                <p class="prediction-label" style="color: {color};">{prediction}</p>
                <p class="{'confidence-high' if confidence > 0.7 else 'confidence-low'}">
                    {confidence*100:.1f}%
                </p>
                <p style="color: #888; margin-top: 0.5rem;">confidence</p>
                <p style="margin-top: 1rem;">{class_descriptions[prediction]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            st.markdown("### Per-class probability")
            fig = create_probability_chart(result['probabilities'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Grad-CAM visualization
            if result['gradcam'] is not None and st.session_state.current_processed_img is not None:
                st.markdown("### Grad-CAM model focus")
                
                gradcam_overlay = create_gradcam_overlay(
                    st.session_state.current_processed_img, 
                    result['gradcam']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.current_processed_img, caption="Original MRI", use_container_width=True)
                with col2:
                    st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)
                
                # Store for PDF
                st.session_state.gradcam_overlay = gradcam_overlay
        else:
            st.info("👆 Upload an MRI scan or select a sample to see results")
    
    # Download report section (below both columns)
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown("### Download report")
        st.caption("The PDF includes the uploaded image, Grad-CAM heatmap, prediction, per-class probabilities, timestamp, and the disclaimer.")
        
        if st.button("📄 Download PDF Report", use_container_width=True):
            result = st.session_state.prediction_result
            image = st.session_state.current_image
            
            if 'gradcam_overlay' in st.session_state:
                pdf_buffer = generate_pdf_report(
                    image,
                    result['prediction'],
                    result['confidence'],
                    result['probabilities'],
                    st.session_state.gradcam_overlay
                )
                
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name=f"brain_mri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚠️ Disclaimer: This tool is for research purposes only. Not for clinical diagnosis.</p>
        <p>© 2024 Brain MRI Tumor Classifier | Powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
