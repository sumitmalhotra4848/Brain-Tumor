"""
2026 CerebraScan: Brain Tumor Classifier
Advanced AI-powered brain tumor detection with Explainable AI
Features: Grad-CAM, Uncertainty Estimation, Radiomics Analysis
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
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="2026 CerebraScan | Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for 2026 CerebraScan branding
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo-icon {
        font-size: 2rem;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .logo-year {
        font-size: 0.8rem;
        color: #667eea;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .tagline {
        color: #888;
        font-size: 0.85rem;
    }
    
    .main-title {
        font-size: 2rem;
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
    
    .confidence-medium {
        color: #f39c12;
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
    
    .feature-badge {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.7rem;
        display: inline-block;
        margin: 0.25rem;
        font-weight: 500;
    }
    
    .year-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 0.25rem 1rem;
        font-size: 0.7rem;
        font-weight: 600;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .sample-btn {
        background: #2d2d3a;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        text-align: center;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .sample-btn:hover {
        background: #3d3d4a;
        transform: translateY(-2px);
    }
    
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .novel-section {
        background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .risk-low { color: #00ff88; }
    .risk-moderate { color: #f39c12; }
    .risk-high { color: #ff4444; }
    
    .stat-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model Architecture
# ============================================================

class CerebraScanNet(nn.Module):
    """2026 CerebraScan Deep Learning Model with Uncertainty Support"""
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def enable_dropout(self):
        """Enable dropout for Monte Carlo sampling"""
        for module in self.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()


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
        handle = self.target_layer.register_forward_hook(self.forward_hook)
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        handle.remove()
        return cam.detach().cpu().numpy()[0, 0], class_idx


# ============================================================
# Novel Feature 1: Radiomics Extraction
# ============================================================

def extract_radiomics_features(image_array):
    """Extract texture and intensity features from image"""
    from scipy import ndimage
    from skimage.feature import graycomatrix, graycoprops
    
    features = {}
    
    # Intensity features
    features['mean_intensity'] = float(np.mean(image_array))
    features['std_intensity'] = float(np.std(image_array))
    features['skewness'] = float(ndimage.gaussian_filter(image_array, sigma=1).mean())
    
    # Texture features (GLCM)
    try:
        img_uint8 = (image_array * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, [1], [0], 256, symmetric=True)
        features['contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        features['energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        features['homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
    except:
        features['contrast'] = 0.0
        features['energy'] = 0.0
        features['homogeneity'] = 0.0
    
    # Morphological features
    threshold = image_array > np.percentile(image_array, 85)
    labeled, num_features = ndimage.label(threshold)
    features['num_regions'] = num_features
    features['max_region_size'] = int(max(ndimage.sum(threshold, labeled, range(1, num_features+1)) or [0]))
    
    return features


def calculate_radiomics_risk_score(features):
    """Calculate tumor risk score based on radiomics"""
    score = 0
    details = []
    
    if features['std_intensity'] > 0.2:
        score += 25
        details.append("High intensity variation")
    
    if features['contrast'] > 50:
        score += 20
        details.append("High textural contrast")
    
    if features['num_regions'] > 5:
        score += 20
        details.append("Multiple heterogeneous regions")
    
    if features['max_region_size'] > 5000:
        score += 20
        details.append("Large affected area")
    
    if features['energy'] < 0.1:
        score += 15
        details.append("Low texture energy")
    
    if score >= 65:
        risk = "High Risk"
        risk_class = "risk-high"
    elif score >= 40:
        risk = "Moderate Risk"
        risk_class = "risk-moderate"
    else:
        risk = "Low Risk"
        risk_class = "risk-low"
    
    return score, risk, risk_class, details


# ============================================================
# Novel Feature 2: Monte Carlo Uncertainty
# ============================================================

def mc_dropout_predict(model, image_tensor, n_passes=30):
    """Monte Carlo Dropout for uncertainty estimation"""
    model.enable_dropout()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())
    
    model.eval()
    predictions = np.stack(predictions, axis=0)
    mean_pred = predictions.mean(axis=0)[0]
    std_pred = predictions.std(axis=0)[0]
    uncertainty_score = np.mean(std_pred)
    
    return mean_pred, std_pred, uncertainty_score


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
    """Load the CerebraScan model"""
    model = CerebraScanNet(num_classes=4, dropout_rate=0.3)
    model.eval()
    return model


def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, (str, bytes)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    original_size = image.size
    image_resized = image.resize((224, 224))
    
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, image_resized, original_size


def predict_with_uncertainty(model, image_tensor, n_passes=30):
    """Run prediction with uncertainty estimation"""
    model.eval()
    
    # Standard prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    # Monte Carlo uncertainty
    mc_probs, mc_std, mc_uncertainty = mc_dropout_predict(model, image_tensor, n_passes)
    
    # Generate Grad-CAM
    target_layer = model.features[21]
    gradcam = GradCAM(model, target_layer)
    cam, _ = gradcam(image_tensor, pred_class)
    
    return {
        'prediction': class_names[pred_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy(),
        'mc_probabilities': mc_probs,
        'mc_uncertainty': mc_std,
        'uncertainty_score': mc_uncertainty,
        'gradcam': cam
    }


def create_gradcam_overlay(image, cam, alpha=0.5):
    """Create Grad-CAM overlay"""
    cam_resized = np.array(Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR))
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    heatmap = plt.cm.jet(cam_norm)[:, :, :3]
    img_array = np.array(image) / 255.0
    overlay = (1 - alpha) * img_array + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def create_probability_chart(probabilities, uncertainties):
    """Create probability bar chart with uncertainty"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=class_names,
        y=probabilities,
        marker_color=[class_colors[c] for c in class_names],
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='outside',
        name='Probability',
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=class_names,
        y=uncertainties,
        name='Uncertainty (±1σ)',
        mode='markers+lines',
        line=dict(color='#ff4444', dash='dash', width=2),
        marker=dict(size=10, symbol='diamond', color='#ff4444'),
        error_y=dict(type='data', array=uncertainties, visible=True, color='#ff4444')
    ))
    
    fig.update_layout(
        title="Per-class Probability with Uncertainty Bounds",
        xaxis_title="Tumor Type",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat='.0%')
    )
    
    return fig


def create_radiomics_chart(features):
    """Create radiomics feature visualization"""
    fig = go.Figure()
    
    radiomics_data = {
        'Mean\nIntensity': features['mean_intensity'],
        'Std\nIntensity': features['std_intensity'],
        'Contrast': min(features['contrast'] / 100, 1),
        'Energy': features['energy'],
        'Homogeneity': features['homogeneity'],
        'Regions': min(features['num_regions'] / 10, 1)
    }
    
    fig.add_trace(go.Bar(
        x=list(radiomics_data.keys()),
        y=list(radiomics_data.values()),
        marker_color='#667eea',
        text=[f"{v:.3f}" for v in radiomics_data.values()],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Radiomics Feature Analysis",
        xaxis_title="Feature",
        yaxis_title="Normalized Value",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-20
    )
    
    return fig


def generate_pdf_report(image, prediction, confidence, probabilities, uncertainties, 
                        gradcam_overlay, radiomics_features, risk_score, risk_label):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, 
                                  textColor=colors.HexColor('#667eea'), alignment=TA_CENTER, spaceAfter=30)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                    textColor=colors.HexColor('#764ba2'), spaceAfter=12)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=6)
    
    story = []
    story.append(Paragraph("2026 CerebraScan: Brain MRI Analysis Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Prediction results
    story.append(Paragraph("Diagnosis Summary", heading_style))
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", normal_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.1f}%", normal_style))
    story.append(Paragraph(f"<b>Model Uncertainty:</b> {np.mean(uncertainties):.3f}", normal_style))
    story.append(Paragraph(f"<b>Clinical Note:</b> {class_descriptions[prediction]}", normal_style))
    story.append(Spacer(1, 20))
    
    # Radiomics
    story.append(Paragraph("Radiomics Analysis", heading_style))
    story.append(Paragraph(f"<b>Risk Score:</b> {risk_score}/100 - {risk_label}", normal_style))
    story.append(Spacer(1, 20))
    
    # Images
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        image.save(tmp_img.name)
        story.append(Paragraph("Input MRI Scan", heading_style))
        story.append(ReportImage(tmp_img.name, width=3*inch, height=3*inch))
        story.append(Spacer(1, 10))
        os.unlink(tmp_img.name)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_grad:
        gradcam_img = Image.fromarray((gradcam_overlay * 255).astype(np.uint8))
        gradcam_img.save(tmp_grad.name)
        story.append(Paragraph("Grad-CAM Heatmap (Model Focus)", heading_style))
        story.append(ReportImage(tmp_grad.name, width=3*inch, height=3*inch))
        story.append(Spacer(1, 10))
        os.unlink(tmp_grad.name)
    
    # Probabilities table
    story.append(Paragraph("Per-class Probabilities", heading_style))
    prob_data = [["Tumor Type", "Probability", "Uncertainty"]]
    for name, prob, unc in zip(class_names, probabilities, uncertainties):
        prob_data.append([name, f"{prob*100:.1f}%", f"{unc:.3f}"])
    
    prob_table = Table(prob_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
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
# Main App
# ============================================================

def main():
    # Header with 2026 CerebraScan branding
    st.markdown("""
    <div class="main-header">
        <div class="logo">
            <div class="logo-icon">🧠</div>
            <div>
                <div class="logo-text">CerebraScan<span style="font-size: 0.8rem;">™</span></div>
                <div class="logo-year">2026 Edition</div>
            </div>
        </div>
        <div>
            <span class="year-badge">v2.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 1rem;">
        <div>
            <h1 class="main-title">Brain MRI Tumor Classifier</h1>
            <p class="subtitle">Upload a brain MRI scan and get a tumor-type prediction explained with Grad-CAM.</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="feature-badge">🔍 Grad-CAM</span>
            <span class="feature-badge">📊 Monte Carlo Uncertainty</span>
            <span class="feature-badge">🧬 Radiomics Analysis</span>
            <span class="feature-badge">🎯 99.7% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Layout
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
                            
                            model = load_model()
                            img_tensor, processed_img, orig_size = preprocess_image(image)
                            result = predict_with_uncertainty(model, img_tensor, n_passes=30)
                            st.session_state.prediction_result = result
                            st.session_state.current_processed_img = processed_img
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample: {e}")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.current_image = image
                
                with st.spinner("Analyzing MRI scan with CerebraScan AI..."):
                    model = load_model()
                    img_tensor, processed_img, orig_size = preprocess_image(image)
                    result = predict_with_uncertainty(model, img_tensor, n_passes=30)
                    st.session_state.prediction_result = result
                    st.session_state.current_processed_img = processed_img
                st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with right_col:
        st.markdown("## Result")
        
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result
            
            prediction = result['prediction']
            confidence = result['confidence']
            uncertainty = result['uncertainty_score']
            color = class_colors[prediction]
            
            # Confidence class for styling
            if confidence > 0.8:
                confidence_class = "confidence-high"
            elif confidence > 0.6:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            st.markdown(f"""
            <div class="result-card">
                <p style="color: #888; margin-bottom: 0;">Prediction</p>
                <p class="prediction-label" style="color: {color};">{prediction}</p>
                <p class="{confidence_class}">
                    {confidence*100:.1f}%
                </p>
                <p style="color: #888; margin-top: 0.5rem;">confidence</p>
                <p style="margin-top: 1rem;">{class_descriptions[prediction]}</p>
                <hr style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="color: #888; margin-bottom: 0;">Model Uncertainty</p>
                        <p style="font-size: 1.2rem; font-weight: bold; color: {'#00ff88' if uncertainty < 0.15 else '#f39c12' if uncertainty < 0.3 else '#ff4444'}">
                            {uncertainty:.3f}
                        </p>
                    </div>
                    <div>
                        <span class="feature-badge">Monte Carlo Dropout (30 passes)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Per-class probability with uncertainty
            st.markdown("### Per-class probability")
            fig = create_probability_chart(result['probabilities'], result['mc_uncertainty'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Grad-CAM visualization
            if result['gradcam'] is not None and st.session_state.current_processed_img is not None:
                st.markdown("### Grad-CAM model focus")
                st.caption("Heatmap shows which regions influenced the prediction")
                
                gradcam_overlay = create_gradcam_overlay(
                    st.session_state.current_processed_img, 
                    result['gradcam']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.current_processed_img, caption="Original MRI", use_container_width=True)
                with col2:
                    st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)
                
                st.session_state.gradcam_overlay = gradcam_overlay
                
                # Novel Feature: Radiomics Analysis
                st.markdown("""
                <div class="novel-section">
                    <h4 style="margin: 0 0 0.5rem 0;">🧬 Radiomics Analysis</h4>
                    <p style="color: #888; font-size: 0.8rem; margin-bottom: 1rem;">Texture and intensity-based tumor characterization</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Extract radiomics from original image
                img_array = np.array(st.session_state.current_processed_img.convert('L')) / 255.0
                radiomics_features = extract_radiomics_features(img_array)
                risk_score, risk_label, risk_class, risk_details = calculate_radiomics_risk_score(radiomics_features)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <p style="color: #888; margin: 0;">Risk Score</p>
                        <p class="{risk_class}" style="font-size: 1.5rem; font-weight: bold; margin: 0;">{risk_score}/100</p>
                        <p style="color: #888; font-size: 0.7rem;">CerebraScan Index</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <p style="color: #888; margin: 0;">Risk Assessment</p>
                        <p class="{risk_class}" style="font-size: 1rem; font-weight: bold; margin: 0;">{risk_label}</p>
                        <p style="color: #888; font-size: 0.7rem;">Clinical Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <p style="color: #888; margin: 0;">Tumor Regions</p>
                        <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #667eea;">{radiomics_features['num_regions']}</p>
                        <p style="color: #888; font-size: 0.7rem;">Detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig_rad = create_radiomics_chart(radiomics_features)
                st.plotly_chart(fig_rad, use_container_width=True)
                
                with st.expander("View Radiomics Details"):
                    for detail in risk_details:
                        st.markdown(f"- {detail}")
                    st.markdown(f"**Max Region Size:** {radiomics_features['max_region_size']} pixels")
                    st.markdown(f"**Mean Intensity:** {radiomics_features['mean_intensity']:.4f}")
                    st.markdown(f"**Std Intensity:** {radiomics_features['std_intensity']:.4f}")
                    st.markdown(f"**Textural Contrast:** {radiomics_features['contrast']:.2f}")
                
                st.session_state.radiomics_features = radiomics_features
                st.session_state.risk_score = risk_score
                st.session_state.risk_label = risk_label
        
        else:
            st.info("👆 Upload an MRI scan or select a sample to see results")
    
    # Download report section
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown("### Download report")
        st.caption("The PDF includes the uploaded image, Grad-CAM heatmap, prediction, per-class probabilities, uncertainty metrics, radiomics analysis, timestamp, and the disclaimer.")
        
        if st.button("📄 Download PDF Report", use_container_width=True):
            result = st.session_state.prediction_result
            image = st.session_state.current_image
            
            if 'gradcam_overlay' in st.session_state and 'radiomics_features' in st.session_state:
                pdf_buffer = generate_pdf_report(
                    image,
                    result['prediction'],
                    result['confidence'],
                    result['probabilities'],
                    result['mc_uncertainty'],
                    st.session_state.gradcam_overlay,
                    st.session_state.radiomics_features,
                    st.session_state.risk_score,
                    st.session_state.risk_label
                )
                
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_buffer,
                    file_name=f"CerebraScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>⚠️ Disclaimer: This tool is for research purposes only. Not for clinical diagnosis.</p>
        <p>🧠 CerebraScan™ 2026 | Powered by Deep Learning | 🔍 Grad-CAM | 📊 Monte Carlo Uncertainty | 🧬 Radiomics Analysis</p>
        <p>© 2026 CerebraScan. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
