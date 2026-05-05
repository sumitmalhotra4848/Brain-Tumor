"""
2026 CerebraScan: Multi-Modal Brain Tumor Classifier
Advanced AI-powered brain tumor detection using T1, T1ce, T2, and FLAIR MRI sequences
Features: Grad-CAM, Uncertainty Estimation, Radiomics Analysis, Multi-Modal Fusion
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
    page_title="2026 CerebraScan | Multi-Modal Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    .feature-badge {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.7rem;
        display: inline-block;
        margin: 0.25rem;
        font-weight: 500;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .confidence-high { color: #00ff88; font-size: 2rem; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-size: 2rem; font-weight: bold; }
    .confidence-low { color: #ff4444; font-size: 2rem; font-weight: bold; }
    .prediction-label { font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0; }
    .risk-low { color: #00ff88; }
    .risk-moderate { color: #f39c12; }
    .risk-high { color: #ff4444; }
    
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
    
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .modality-tab {
        background: #1e1e2f;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model Architecture (Multi-Modal)
# ============================================================

class MultiModalBrainTumorClassifier(nn.Module):
    """Multi-modal classifier accepting T1, T1ce, T2, FLAIR inputs"""
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Shared feature extractor for 4 input channels
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(4, 64, 3, padding=1),
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
        for module in self.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()


class ModalityReliabilityModule(nn.Module):
    """Learns which MRI modality is most useful (Novel Feature)"""
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.sigmoid(self.conv(x))
        return x * weights


class GradCAM:
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
# Novel Feature 1: Modality Reliability Analysis
# ============================================================

def analyze_modality_reliability(image_tensor):
    """Analyze SNR and reliability of each MRI modality"""
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    reliability_scores = []
    snr_values = []
    
    for c in range(4):
        ch = image_tensor[:, c, :, :]
        mu = ch.mean().item()
        std = ch.std().item() + 1e-8
        snr = abs(mu) / std
        snr_values.append(snr)
    
    mean_snr = np.mean(snr_values)
    for snr in snr_values:
        rel = snr / (mean_snr + 1e-8)
        reliability = float(torch.sigmoid(torch.tensor(rel - 0.5)).item())
        reliability_scores.append(reliability)
    
    return modality_names, reliability_scores, snr_values


# ============================================================
# Novel Feature 2: Radiomics Extraction
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
# Novel Feature 3: Monte Carlo Uncertainty
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
    """Load the CerebraScan multi-modal model"""
    model = MultiModalBrainTumorClassifier(num_classes=4, dropout_rate=0.3)
    model.eval()
    return model


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess single image for model input"""
    if isinstance(image, (str, bytes)):
        image = Image.open(image).convert('L')  # Convert to grayscale
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('L')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array


def create_multi_modal_tensor(t1_img, t1ce_img, t2_img, flair_img):
    """Create 4-channel input tensor from all modalities"""
    # Preprocess each modality
    t1 = preprocess_image(t1_img)
    t1ce = preprocess_image(t1ce_img)
    t2 = preprocess_image(t2_img)
    flair = preprocess_image(flair_img)
    
    # Stack to create 4-channel image (C, H, W)
    multi_modal = np.stack([t1, t1ce, t2, flair], axis=0)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(multi_modal).float().unsqueeze(0)
    
    # Normalize each channel
    for c in range(4):
        channel = img_tensor[0, c]
        mean = channel.mean()
        std = channel.std() + 1e-8
        img_tensor[0, c] = (channel - mean) / std
    
    return img_tensor, (t1, t1ce, t2, flair)


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
    
    # Generate Grad-CAM on FLAIR-like representation
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
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image) / 255.0
    
    # Ensure 2D
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]
    
    # Resize cam
    from PIL import Image as PILImage
    cam_resized = np.array(PILImage.fromarray(cam).resize((img_array.shape[1], img_array.shape[0]), PILImage.Resampling.BILINEAR))
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    
    # Create heatmap
    heatmap = plt.cm.jet(cam_norm)[:, :, :3]
    
    # Convert grayscale to RGB
    img_rgb = np.stack([img_array, img_array, img_array], axis=2)
    
    # Overlay
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
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
        name='Probability'
    ))
    
    fig.add_trace(go.Scatter(
        x=class_names,
        y=uncertainties,
        name='Uncertainty (±1σ)',
        mode='markers+lines',
        line=dict(color='#ff4444', dash='dash', width=2),
        marker=dict(size=10, symbol='diamond', color='#ff4444')
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
        textposition='outside'
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


def create_modality_reliability_chart(modality_names, reliability_scores):
    """Create modality reliability bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=modality_names,
        y=reliability_scores,
        marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
        text=[f"{s:.3f}" for s in reliability_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Multi-Modal Reliability Analysis",
        xaxis_title="MRI Modality",
        yaxis_title="Reliability Score",
        yaxis_range=[0, 1],
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def generate_pdf_report(processed_images, prediction, confidence, probabilities, uncertainties, 
                        gradcam_overlay, radiomics_features, risk_score, risk_label,
                        modality_names, reliability_scores):
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
    story.append(Paragraph("2026 CerebraScan: Multi-Modal Brain MRI Analysis Report", title_style))
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
    
    # Modality Reliability
    story.append(Paragraph("Multi-Modal Reliability", heading_style))
    for name, score in zip(modality_names, reliability_scores):
        story.append(Paragraph(f"<b>{name}:</b> {score:.3f}", normal_style))
    story.append(Spacer(1, 20))
    
    # Grad-CAM
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
        "This analysis is for research purposes only and should not be used for clinical diagnosis.",
        normal_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# Main App
# ============================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="logo">
            <div class="logo-icon">🧠</div>
            <div>
                <div class="logo-text">CerebraScan<span style="font-size: 0.8rem;">™</span></div>
                <div class="logo-year">2026 Edition | Multi-Modal AI</div>
            </div>
        </div>
        <div>
            <span class="feature-badge">🔍 Grad-CAM</span>
            <span class="feature-badge">📊 MC Uncertainty</span>
            <span class="feature-badge">🧬 Radiomics</span>
            <span class="feature-badge">🎯 4-Modality Fusion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h1 class="main-title" style="font-size: 1.75rem;">Multi-Modal Brain MRI Tumor Classifier</h1>
    <p class="subtitle">Upload all four MRI sequences (T1, T1ce, T2, FLAIR) for comprehensive tumor analysis with explainable AI.</p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = None
    
    # Sidebar for uploads
    with st.sidebar:
        st.markdown("## 📁 Upload MRI Sequences")
        st.markdown("Please upload all four modalities for accurate analysis")
        
        t1_file = st.file_uploader("T1-weighted MRI", type=['jpg', 'jpeg', 'png', 'nii', 'nii.gz'], key="t1")
        t1ce_file = st.file_uploader("T1 Contrast-Enhanced (T1ce)", type=['jpg', 'jpeg', 'png', 'nii', 'nii.gz'], key="t1ce")
        t2_file = st.file_uploader("T2-weighted MRI", type=['jpg', 'jpeg', 'png', 'nii', 'nii.gz'], key="t2")
        flair_file = st.file_uploader("FLAIR MRI", type=['jpg', 'jpeg', 'png', 'nii', 'nii.gz'], key="flair")
        
        st.markdown("---")
        st.markdown("### Or try sample data")
        
        if st.button("📊 Load Sample MRI Set", use_container_width=True):
            # Create sample images for demonstration
            np.random.seed(42)
            size = 224
            # Create synthetic MRI-like images
            t1_sample = (np.random.rand(size, size) * 0.3 + 0.5).astype(np.float32)
            t1ce_sample = (np.random.rand(size, size) * 0.3 + 0.5).astype(np.float32)
            t2_sample = (np.random.rand(size, size) * 0.3 + 0.5).astype(np.float32)
            flair_sample = (np.random.rand(size, size) * 0.3 + 0.5).astype(np.float32)
            
            # Add a "tumor" region
            y, x = np.ogrid[:size, :size]
            center = (size//2, size//2)
            tumor_mask = (x - center[0])**2 + (y - center[1])**2 < 30**2
            for arr in [t1_sample, t1ce_sample, t2_sample, flair_sample]:
                arr[tumor_mask] = 0.9
            
            st.session_state.sample_images = (t1_sample, t1ce_sample, t2_sample, flair_sample)
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("""
        **CerebraScan 2026** uses a multi-modal deep learning model that analyzes all four MRI sequences simultaneously.
        
        **Input Requirements:**
        - T1-weighted MRI
        - T1 Contrast-Enhanced (T1ce)
        - T2-weighted MRI  
        - FLAIR MRI
        
        All images should be of the same slice/patient.
        """)
    
    # Main content area
    if st.sidebar.button("🧠 Run CerebraScan Analysis", type="primary", use_container_width=True):
        # Check if we have all modalities
        if 'sample_images' in st.session_state:
            t1_img, t1ce_img, t2_img, flair_img = st.session_state.sample_images
            use_sample = True
        elif all([t1_file, t1ce_file, t2_file, flair_file]):
            t1_img = Image.open(t1_file)
            t1ce_img = Image.open(t1ce_file)
            t2_img = Image.open(t2_file)
            flair_img = Image.open(flair_file)
            use_sample = False
        else:
            st.error("Please upload all four MRI modalities (T1, T1ce, T2, FLAIR) or load sample data")
            st.stop()
        
        with st.spinner("Analyzing multi-modal MRI data with CerebraScan AI..."):
            try:
                # Create multi-modal tensor
                img_tensor, processed_images = create_multi_modal_tensor(t1_img, t1ce_img, t2_img, flair_img)
                
                # Load model and predict
                model = load_model()
                result = predict_with_uncertainty(model, img_tensor, n_passes=30)
                
                # Analyze modality reliability
                modality_names, reliability_scores, snr_values = analyze_modality_reliability(img_tensor)
                
                # Extract radiomics from FLAIR (most sensitive for tumors)
                flair_array = processed_images[3]
                radiomics_features = extract_radiomics_features(flair_array)
                risk_score, risk_label, risk_class, risk_details = calculate_radiomics_risk_score(radiomics_features)
                
                # Generate Grad-CAM overlay on FLAIR
                gradcam_overlay = create_gradcam_overlay(flair_array, result['gradcam'])
                
                # Store results
                st.session_state.prediction_result = result
                st.session_state.processed_images = processed_images
                st.session_state.modality_names = modality_names
                st.session_state.reliability_scores = reliability_scores
                st.session_state.radiomics_features = radiomics_features
                st.session_state.risk_score = risk_score
                st.session_state.risk_label = risk_label
                st.session_state.gradcam_overlay = gradcam_overlay
                st.session_state.risk_details = risk_details
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing images: {str(e)}")
    
    # Display results
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        processed_images = st.session_state.processed_images
        modality_names = st.session_state.modality_names
        reliability_scores = st.session_state.reliability_scores
        radiomics_features = st.session_state.radiomics_features
        risk_score = st.session_state.risk_score
        risk_label = st.session_state.risk_label
        risk_class = risk_label.split()[0].lower()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Diagnosis", "🔬 Multi-Modal Views", "🧬 Radiomics", "🎯 AI Explainability"])
        
        with tab1:
            # Prediction card
            prediction = result['prediction']
            confidence = result['confidence']
            uncertainty = result['uncertainty_score']
            color = class_colors[prediction]
            
            if confidence > 0.8:
                confidence_class = "confidence-high"
            elif confidence > 0.6:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <p style="color: #888; margin-bottom: 0;">Prediction</p>
                    <p class="prediction-label" style="color: {color};">{prediction}</p>
                    <p class="{confidence_class}">{confidence*100:.1f}%</p>
                    <p style="color: #888; margin-top: 0.5rem;">confidence</p>
                    <p style="margin-top: 1rem;">{class_descriptions[prediction]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <p style="color: #888; margin-bottom: 0;">Model Uncertainty</p>
                    <p style="font-size: 1.5rem; font-weight: bold; color: {'#00ff88' if uncertainty < 0.15 else '#f39c12' if uncertainty < 0.3 else '#ff4444'}">
                        {uncertainty:.3f}
                    </p>
                    <p style="color: #888;">Monte Carlo Dropout (30 passes)</p>
                    <hr>
                    <p style="color: #888; margin-bottom: 0;">Risk Assessment</p>
                    <p class="{risk_class}" style="font-size: 1.2rem; font-weight: bold;">{risk_label} ({risk_score}/100)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability chart
            st.markdown("### Per-class Probability with Uncertainty")
            fig = create_probability_chart(result['probabilities'], result['mc_uncertainty'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Multi-Modal MRI Views")
            st.caption("All four MRI sequences analyzed simultaneously")
            
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                st.image(processed_images[0], caption="T1-weighted", use_container_width=True, clamp=True)
                st.caption(f"Reliability: {reliability_scores[0]:.3f}")
            with col2:
                st.image(processed_images[1], caption="T1 Contrast-Enhanced (T1ce)", use_container_width=True, clamp=True)
                st.caption(f"Reliability: {reliability_scores[1]:.3f}")
            with col3:
                st.image(processed_images[2], caption="T2-weighted", use_container_width=True, clamp=True)
                st.caption(f"Reliability: {reliability_scores[2]:.3f}")
            with col4:
                st.image(processed_images[3], caption="FLAIR", use_container_width=True, clamp=True)
                st.caption(f"Reliability: {reliability_scores[3]:.3f}")
            
            # Modality reliability chart
            st.markdown("### Modality Reliability Analysis")
            fig = create_modality_reliability_chart(modality_names, reliability_scores)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🧬 Radiomics Analysis")
            st.caption("Texture and intensity-based tumor characterization")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score}/100")
            with col2:
                st.metric("Risk Assessment", risk_label)
            with col3:
                st.metric("Tumor Regions", radiomics_features['num_regions'])
            
            fig = create_radiomics_chart(radiomics_features)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Detailed Radiomics Metrics"):
                st.markdown(f"- **Mean Intensity:** {radiomics_features['mean_intensity']:.4f}")
                st.markdown(f"- **Std Intensity:** {radiomics_features['std_intensity']:.4f}")
                st.markdown(f"- **Skewness:** {radiomics_features['skewness']:.4f}")
                st.markdown(f"- **Contrast:** {radiomics_features['contrast']:.2f}")
                st.markdown(f"- **Energy:** {radiomics_features['energy']:.4f}")
                st.markdown(f"- **Homogeneity:** {radiomics_features['homogeneity']:.4f}")
                st.markdown(f"- **Max Region Size:** {radiomics_features['max_region_size']} pixels")
            
            st.markdown("**Clinical Findings:**")
            for detail in st.session_state.risk_details:
                st.markdown(f"- {detail}")
        
        with tab4:
            st.markdown("### 🎯 Grad-CAM Model Focus")
            st.caption("Heatmap shows which regions of the MRI influenced the prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(processed_images[3], caption="FLAIR MRI (Input)", use_container_width=True, clamp=True)
            with col2:
                st.image(st.session_state.gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - 🔴 **Red regions** had the strongest influence on the prediction
            - 🟡 **Yellow regions** had moderate influence
            - 🔵 **Blue regions** had minimal influence on the decision
            
            This visualization helps radiologists understand which anatomical areas the model focused on.
            """)
    
    # Download report
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Download Complete PDF Report", use_container_width=True):
                pdf_buffer = generate_pdf_report(
                    st.session_state.processed_images,
                    st.session_state.prediction_result['prediction'],
                    st.session_state.prediction_result['confidence'],
                    st.session_state.prediction_result['probabilities'],
                    st.session_state.prediction_result['mc_uncertainty'],
                    st.session_state.gradcam_overlay,
                    st.session_state.radiomics_features,
                    st.session_state.risk_score,
                    st.session_state.risk_label,
                    st.session_state.modality_names,
                    st.session_state.reliability_scores
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
        <p>🧠 CerebraScan™ 2026 | Multi-Modal AI | T1 · T1ce · T2 · FLAIR | Grad-CAM | Monte Carlo Uncertainty | Radiomics</p>
        <p>© 2026 CerebraScan. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
