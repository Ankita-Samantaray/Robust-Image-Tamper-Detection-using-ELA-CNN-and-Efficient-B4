#!/usr/bin/env python3
"""
Generate comprehensive PDF report for FakeImageDetector project
Includes: What, Why, How, and Architecture Diagrams
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage, Preformatted
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
from io import BytesIO

# Create diagrams directory
os.makedirs('report_diagrams', exist_ok=True)

# Report content
def create_architecture_diagram_ela():
    """Create architecture diagram for ELA+CNN method"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'ELA + CNN Architecture', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    # Input Image
    input_box = FancyBboxPatch((1, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 6.4, 'Input\nImage', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow
    ax.arrow(2.5, 6.4, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # ELA Processing
    ela_box = FancyBboxPatch((3.3, 6), 2, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(ela_box)
    ax.text(4.3, 6.4, 'ELA Processing\n(Error Level Analysis)', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Arrow down
    ax.arrow(4.3, 6, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Resize and Normalize
    resize_box = FancyBboxPatch((3.3, 4.5), 2, 0.5, boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(resize_box)
    ax.text(4.3, 4.75, 'Resize to 128x128\nNormalize (/255)', ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.arrow(4.3, 4.5, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # CNN Architecture
    cnn_box = FancyBboxPatch((1, 3), 8, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(5, 3.4, 'CNN Model', ha='center', va='center', fontsize=12, weight='bold')
    
    # CNN Layers details
    layers = [
        ('Conv2D\n(32 filters)', 2, 2),
        ('Conv2D\n(64 filters)', 3.5, 2),
        ('Conv2D\n(128 filters)', 5, 2),
        ('MaxPool2D', 6.5, 2),
        ('Dense\n(256)', 8, 2)
    ]
    
    for layer_name, x_pos, y_pos in layers:
        layer_box = FancyBboxPatch((x_pos-0.4, y_pos), 0.8, 0.5, boxstyle="round,pad=0.05", 
                                   facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(layer_box)
        ax.text(x_pos, y_pos+0.25, layer_name, ha='center', va='center', fontsize=7)
        if x_pos < 8:
            ax.arrow(x_pos+0.4, y_pos+0.25, 0.2, 0, head_width=0.08, head_length=0.05, fc='gray', ec='gray')
    
    # Arrow down
    ax.arrow(5, 3, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Output
    output_box = FancyBboxPatch((3.5, 1.5), 3, 0.8, boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.9, 'Output\n(Real/Fake)', ha='center', va='center', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('report_diagrams/ela_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'report_diagrams/ela_architecture.png'

def create_architecture_diagram_ai():
    """Create architecture diagram for EfficientNet AI detection"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'EfficientNet-B4 AI-Generated Image Detection Architecture', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Input Image
    input_box = FancyBboxPatch((1, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 6.4, 'Input\nImage', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow
    ax.arrow(2.5, 6.4, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Preprocessing
    preprocess_box = FancyBboxPatch((3.3, 6), 2, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(4.3, 6.4, 'Preprocessing\nResize to 224x224\nNormalize [0,1]', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Arrow down
    ax.arrow(4.3, 6, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # EfficientNet-B4 Base
    efficient_box = FancyBboxPatch((2, 4), 6, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(efficient_box)
    ax.text(5, 4.4, 'EfficientNet-B4 Base\n(Pre-trained on ImageNet)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Arrow down
    ax.arrow(5, 4, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Custom Head
    head_box = FancyBboxPatch((3, 2.8), 4, 0.6, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(head_box)
    ax.text(5, 3.1, 'Custom Classification Head', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Head details
    head_layers = [
        ('GlobalAvgPool2D', 3.5, 2),
        ('Dense(512)\n+ Dropout(0.5)', 5, 2),
        ('Dense(256)\n+ Dropout(0.3)', 6.5, 2)
    ]
    
    for layer_name, x_pos, y_pos in head_layers:
        layer_box = FancyBboxPatch((x_pos-0.5, y_pos), 1, 0.4, boxstyle="round,pad=0.05", 
                                   facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(layer_box)
        ax.text(x_pos, y_pos+0.2, layer_name, ha='center', va='center', fontsize=7)
        if x_pos < 6.5:
            ax.arrow(x_pos+0.5, y_pos+0.2, 0.3, 0, head_width=0.08, head_length=0.05, fc='gray', ec='gray')
    
    # Arrow down
    ax.arrow(5, 2, 0, -0.3, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Output
    output_box = FancyBboxPatch((3, 1), 4, 0.6, boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.3, 'Output\n(Real/GAN/Diffusion)', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('report_diagrams/ai_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'report_diagrams/ai_architecture.png'

def create_system_overview_diagram():
    """Create overall system overview diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.5, 'FakeImageDetector System Overview', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    # Input
    input_box = FancyBboxPatch((0.5, 4), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 4.5, 'Input Image', ha='center', va='center', fontsize=11, weight='bold')
    
    # Three detection methods
    methods = [
        ('ELA + CNN\nDetection', 4, 4, 'lightgreen', 2.5),
        ('EfficientNet-B4\nAI Detection', 8, 4, 'lightyellow', 2.5),
        ('Metadata\nAnalysis', 6, 2, 'lightcoral', 2)
    ]
    
    for method_name, x, y, color, width in methods:
        method_box = FancyBboxPatch((x-width/2, y), width, 1, boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(method_box)
        ax.text(x, y+0.5, method_name, ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrow from input
        if y == 4:
            ax.arrow(2.5, 4.5, x-width/2-2.5, 0, head_width=0.15, head_length=0.1, 
                    fc='black', ec='black')
        else:
            ax.arrow(1.5, 4, x-1.5, y+1-4, head_width=0.15, head_length=0.1, 
                    fc='black', ec='black')
    
    # Output
    output_box = FancyBboxPatch((9.5, 2), 2, 1, boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(10.5, 2.5, 'Detection\nResult', ha='center', va='center', fontsize=11, weight='bold')
    
    # Arrows to output
    ax.arrow(6.5, 2.5, 3, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.arrow(8, 4, 2.5, -1.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.arrow(8, 2.5, 1.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    plt.tight_layout()
    plt.savefig('report_diagrams/system_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'report_diagrams/system_overview.png'

def create_data_flow_diagram():
    """Create data flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Data Flow and Training Pipeline', ha='center', va='center', 
            fontsize=16, weight='bold')
    
    # Dataset
    dataset_box = FancyBboxPatch((1, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                                 facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(2, 8, 'Dataset\n(Real/Fake/AI)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow
    ax.arrow(3, 8, 1, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Preprocessing
    prep_box = FancyBboxPatch((4, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(prep_box)
    ax.text(5, 8, 'Preprocessing\n(ELA/Resize)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrow down
    ax.arrow(5, 7.5, 0, -1, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Train/Val Split
    split_box = FancyBboxPatch((4, 6), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(split_box)
    ax.text(5, 6.4, 'Train/Val Split\n(80/20)', ha='center', va='center', fontsize=10)
    
    # Arrow down
    ax.arrow(5, 6, 0, -1, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Model Training
    train_box = FancyBboxPatch((2, 4), 6, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(5, 4.5, 'Model Training\n(CNN/EfficientNet)', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Callbacks
    callback_text = 'Callbacks:\n• ModelCheckpoint\n• EarlyStopping\n• ReduceLROnPlateau'
    ax.text(5, 3.5, callback_text, ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow down
    ax.arrow(5, 4, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Model Evaluation
    eval_box = FancyBboxPatch((3, 2.5), 4, 0.7, boxstyle="round,pad=0.1", 
                              facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(5, 2.85, 'Model Evaluation', ha='center', va='center', fontsize=10, weight='bold')
    
    # Metrics
    metrics_text = '• Accuracy\n• Precision/Recall\n• Confusion Matrix'
    ax.text(5, 1.8, metrics_text, ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow down
    ax.arrow(5, 2.5, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # Saved Model
    save_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(save_box)
    ax.text(5, 0.9, 'Saved Model\n(.keras file)', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('report_diagrams/data_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'report_diagrams/data_flow.png'

def create_pdf_report():
    """Create the PDF report"""
    filename = 'FakeImageDetector_Report.pdf'
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("FakeImageDetector", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Comprehensive Project Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", 
                          styles['Normal']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading1_style))
    story.append(Spacer(1, 0.2*inch))
    toc_items = [
        "1. Executive Summary",
        "2. What is FakeImageDetector?",
        "3. Why FakeImageDetector?",
        "4. How Does It Work?",
        "5. Architecture",
        "6. Dataset",
        "7. Training Process",
        "8. Results and Performance",
        "9. Project Structure",
        "10. Conclusion"
    ]
    for item in toc_items:
        story.append(Paragraph(item, body_style))
    story.append(PageBreak())
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading1_style))
    story.append(Paragraph(
        """FakeImageDetector is a comprehensive deep learning system designed to detect 
        fake, tampered, and AI-generated images. The project implements multiple detection 
        methods including Error Level Analysis (ELA) combined with Convolutional Neural Networks 
        (CNN), EfficientNet-B4 based AI-generated image detection, and metadata analysis. 
        This system addresses the growing need for image authenticity verification in the age 
        of advanced image manipulation tools and AI-generated content.""",
        body_style
    ))
    story.append(PageBreak())
    
    # 2. What is FakeImageDetector?
    story.append(Paragraph("2. What is FakeImageDetector?", heading1_style))
    
    story.append(Paragraph(
        """FakeImageDetector is a machine learning project that provides multiple approaches 
        to detect fake and tampered images. The system consists of three main detection methods:""",
        body_style
    ))
    
    story.append(Paragraph("2.1 ELA + CNN Detection Method", heading2_style))
    story.append(Paragraph(
        """This method combines Error Level Analysis (ELA) with Convolutional Neural Networks. 
        ELA is a technique that reveals compression artifacts in images, which are different for 
        original versus edited/tampered images. The ELA-processed images are then fed into a 
        custom CNN architecture for classification as real or fake.""",
        body_style
    ))
    
    story.append(Paragraph("<b>Key Features:</b>", body_style))
    story.append(Paragraph("• Detects image tampering and splicing", body_style))
    story.append(Paragraph("• Uses ELA to highlight compression differences", body_style))
    story.append(Paragraph("• Custom CNN architecture for classification", body_style))
    story.append(Paragraph("• Achieves ~91.83% accuracy", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("2.2 AI-Generated Image Detection", heading2_style))
    story.append(Paragraph(
        """This method uses transfer learning with EfficientNet-B4 to detect AI-generated images. 
        The model can classify images into three categories: Real images, GAN-generated images 
        (StyleGAN, ProGAN), and Diffusion model-generated images (DALL-E, Midjourney, Stable Diffusion).""",
        body_style
    ))
    
    story.append(Paragraph("<b>Key Features:</b>", body_style))
    story.append(Paragraph("• Uses pre-trained EfficientNet-B4 backbone", body_style))
    story.append(Paragraph("• Detects GAN-generated images", body_style))
    story.append(Paragraph("• Detects Diffusion model-generated images", body_style))
    story.append(Paragraph("• Custom classification head for fine-tuning", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("2.3 Metadata Analysis", heading2_style))
    story.append(Paragraph(
        """This component analyzes EXIF metadata from images to detect inconsistencies that 
        may indicate tampering or manipulation. It examines timestamps, creation dates, and 
        other metadata attributes.""",
        body_style
    ))
    
    story.append(PageBreak())
    
    # 3. Why FakeImageDetector?
    story.append(Paragraph("3. Why FakeImageDetector?", heading1_style))
    
    story.append(Paragraph(
        """In today's digital age, image manipulation and AI-generated content have become 
        increasingly sophisticated and prevalent. This creates several critical challenges:""",
        body_style
    ))
    
    story.append(Paragraph("3.1 Problem Statement", heading2_style))
    story.append(Paragraph(
        """<b>Fake News and Misinformation:</b> Manipulated images are often used to spread 
        false information, which can have serious consequences in journalism, social media, 
        and public discourse.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Security Concerns:</b> Fake images can be used for identity theft, fraud, 
        and other malicious activities.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>AI-Generated Content:</b> With the rise of DALL-E, Midjourney, Stable Diffusion, 
        and other AI image generators, it's becoming harder to distinguish between real and 
        synthetic images.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Forensic Applications:</b> Law enforcement and legal proceedings require reliable 
        methods to verify image authenticity.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("3.2 Solution Benefits", heading2_style))
    story.append(Paragraph(
        """FakeImageDetector addresses these challenges by providing:""",
        body_style
    ))
    story.append(Paragraph("• <b>Multiple Detection Methods:</b> Uses different techniques to catch various types of manipulations", body_style))
    story.append(Paragraph("• <b>High Accuracy:</b> Achieves over 90% accuracy in detecting tampered images", body_style))
    story.append(Paragraph("• <b>AI Detection:</b> Specifically designed to detect modern AI-generated content", body_style))
    story.append(Paragraph("• <b>Comprehensive Analysis:</b> Combines visual analysis with metadata examination", body_style))
    story.append(Paragraph("• <b>Open Source:</b> Provides transparency and allows for community contributions", body_style))
    
    story.append(PageBreak())
    
    # 4. How Does It Work?
    story.append(Paragraph("4. How Does It Work?", heading1_style))
    
    story.append(Paragraph("4.1 ELA + CNN Method", heading2_style))
    story.append(Paragraph(
        """<b>Step 1: Error Level Analysis (ELA)</b><br/>
        The original image is saved at a specific JPEG quality (e.g., 90%). The difference 
        between the original and re-saved image is computed. Edited regions show higher 
        error levels, making tampering visible.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Step 2: Preprocessing</b><br/>
        The ELA-processed image is resized to 128×128 pixels and normalized to values between 
        0 and 1.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Step 3: CNN Classification</b><br/>
        The preprocessed image is fed through a CNN with multiple convolutional layers, 
        pooling layers, and dense layers to classify it as real or fake.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.2 AI-Generated Image Detection", heading2_style))
    story.append(Paragraph(
        """<b>Step 1: Image Preprocessing</b><br/>
        Input images are resized to 224×224 pixels (EfficientNet input size) and normalized 
        to [0, 1] range.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Step 2: Feature Extraction</b><br/>
        The EfficientNet-B4 base model (pre-trained on ImageNet) extracts high-level features 
        from the images.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Step 3: Classification</b><br/>
        A custom classification head with dense layers and dropout classifies images into 
        Real, GAN-generated, or Diffusion-generated categories.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("4.3 Metadata Analysis", heading2_style))
    story.append(Paragraph(
        """Extracts and analyzes EXIF metadata including timestamps, camera information, 
        GPS data, and software used. Inconsistencies in metadata can indicate tampering.""",
        body_style
    ))
    
    # Add data flow diagram
    story.append(Spacer(1, 0.3*inch))
    if os.path.exists('report_diagrams/data_flow.png'):
        img = RLImage('report_diagrams/data_flow.png', width=6*inch, height=4.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<i>Figure 1: Data Flow and Training Pipeline</i>", 
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, 
                                           alignment=TA_CENTER, fontStyle='italic')))
    
    story.append(PageBreak())
    
    # 5. Architecture
    story.append(Paragraph("5. Architecture", heading1_style))
    
    story.append(Paragraph("5.1 System Overview", heading2_style))
    story.append(Paragraph(
        """The system consists of three main detection pipelines that can work independently 
        or in combination to provide comprehensive image authenticity verification.""",
        body_style
    ))
    
    # System overview diagram
    if os.path.exists('report_diagrams/system_overview.png'):
        img = RLImage('report_diagrams/system_overview.png', width=6*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Figure 2: System Overview</i>", 
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, 
                                           alignment=TA_CENTER, fontStyle='italic')))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("5.2 ELA + CNN Architecture", heading2_style))
    story.append(Paragraph(
        """The ELA+CNN architecture processes images through error level analysis and then 
        uses a custom CNN for binary classification.""",
        body_style
    ))
    
    # ELA architecture diagram
    if os.path.exists('report_diagrams/ela_architecture.png'):
        img = RLImage('report_diagrams/ela_architecture.png', width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Figure 3: ELA + CNN Architecture</i>", 
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, 
                                           alignment=TA_CENTER, fontStyle='italic')))
    
    story.append(PageBreak())
    
    story.append(Paragraph("5.3 EfficientNet-B4 AI Detection Architecture", heading2_style))
    story.append(Paragraph(
        """The AI detection model uses transfer learning with EfficientNet-B4, a state-of-the-art 
        architecture that balances accuracy and efficiency. The model includes a custom classification 
        head with dropout regularization.""",
        body_style
    ))
    
    # AI architecture diagram
    if os.path.exists('report_diagrams/ai_architecture.png'):
        img = RLImage('report_diagrams/ai_architecture.png', width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Figure 4: EfficientNet-B4 AI Detection Architecture</i>", 
                              ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, 
                                           alignment=TA_CENTER, fontStyle='italic')))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Model Details:</b>", body_style))
    
    model_details = [
        ("Base Model", "EfficientNet-B4 (pre-trained on ImageNet)"),
        ("Input Size", "224×224×3"),
        ("Classification Head", "GlobalAveragePooling2D → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(3)"),
        ("Output Classes", "Real, GAN-generated, Diffusion-generated"),
        ("Optimizer", "Adam (learning rate: 0.0001)"),
        ("Loss Function", "Categorical Crossentropy")
    ]
    
    table_data = [["<b>Component</b>", "<b>Specification</b>"]]
    for detail, spec in model_details:
        table_data.append([detail, spec])
    
    table = Table(table_data, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(table)
    
    story.append(PageBreak())
    
    # 6. Dataset
    story.append(Paragraph("6. Dataset", heading1_style))
    
    story.append(Paragraph("6.1 ELA Detection Dataset", heading2_style))
    story.append(Paragraph(
        """The ELA detection method uses a dataset of approximately 9,500 images split into 
        real and fake categories. Images are processed through ELA before training.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("6.2 AI Detection Dataset", heading2_style))
    story.append(Paragraph(
        """The AI detection dataset consists of three categories:""",
        body_style
    ))
    
    dataset_stats = [
        ("Real Images", "~31,783 images", "Natural photographs from sources like COCO dataset"),
        ("GAN-Generated", "~70,001 images", "Images generated using StyleGAN, ProGAN, etc."),
        ("Diffusion-Generated", "~2,778 images", "Images from DALL-E, Midjourney, Stable Diffusion")
    ]
    
    table_data = [["<b>Category</b>", "<b>Count</b>", "<b>Source</b>"]]
    for cat, count, source in dataset_stats:
        table_data.append([cat, count, source])
    
    table = Table(table_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        """<b>Dataset Structure:</b><br/>
        datasets/<br/>
        ├── ai_detection/<br/>
        │   ├── real/<br/>
        │   ├── gan/<br/>
        │   ├── diffusion/<br/>
        │   └── test/<br/>
        ├── train/<br/>
        │   ├── real/<br/>
        │   └── fake/<br/>
        └── dataset.csv""",
        ParagraphStyle('Code', parent=styles['Code'], fontSize=9, fontName='Courier')
    ))
    
    story.append(PageBreak())
    
    # 7. Training Process
    story.append(Paragraph("7. Training Process", heading1_style))
    
    story.append(Paragraph("7.1 ELA + CNN Training", heading2_style))
    story.append(Paragraph(
        """<b>Configuration:</b><br/>
        • Image size: 128×128×3 (after ELA processing)<br/>
        • Batch size: 100<br/>
        • Epochs: 30 (with early stopping)<br/>
        • Optimizer: RMSprop (learning rate: 0.0005)<br/>
        • Train/Validation split: 80/20""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Training Process:</b><br/>
        1. Load images from CSV file<br/>
        2. Convert each image to ELA format<br/>
        3. Resize and normalize<br/>
        4. Split into training and validation sets<br/>
        5. Train CNN model with early stopping callback<br/>
        6. Save best model based on validation accuracy""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Results:</b> Model converged at epoch 9 with 91.83% accuracy.""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("7.2 AI Detection Training", heading2_style))
    story.append(Paragraph(
        """<b>Configuration:</b><br/>
        • Image size: 224×224×3<br/>
        • Batch size: 32<br/>
        • Epochs: 50 (with early stopping)<br/>
        • Optimizer: Adam (learning rate: 0.0001)<br/>
        • Train/Validation split: 80/20 (stratified)""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Training Process:</b><br/>
        1. Load images from folder structure (real/gan/diffusion)<br/>
        2. Preprocess: resize to 224×224, normalize to [0,1]<br/>
        3. One-hot encode labels for multi-class classification<br/>
        4. Load pre-trained EfficientNet-B4 weights<br/>
        5. Fine-tune with custom classification head<br/>
        6. Use callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau<br/>
        7. Save best model based on validation accuracy""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Callbacks:</b><br/>
        • <b>ModelCheckpoint:</b> Saves best model during training<br/>
        • <b>EarlyStopping:</b> Stops training if no improvement for 5 epochs<br/>
        • <b>ReduceLROnPlateau:</b> Reduces learning rate when validation loss plateaus""",
        body_style
    ))
    
    story.append(PageBreak())
    
    # 8. Results and Performance
    story.append(Paragraph("8. Results and Performance", heading1_style))
    
    story.append(Paragraph("8.1 ELA + CNN Method", heading2_style))
    story.append(Paragraph(
        """The ELA + CNN model achieved excellent performance in detecting tampered images:""",
        body_style
    ))
    story.append(Paragraph(
        """• <b>Best Accuracy:</b> 91.83%<br/>
        • <b>Convergence:</b> Epoch 9<br/>
        • <b>Method:</b> Binary classification (Real vs Fake)""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.2 AI Detection Method", heading2_style))
    story.append(Paragraph(
        """The EfficientNet-B4 model provides multi-class classification for detecting 
        different types of AI-generated content. Performance metrics include:""",
        body_style
    ))
    story.append(Paragraph(
        """• <b>Classification:</b> 3-class (Real, GAN, Diffusion)<br/>
        • <b>Model Size:</b> ~215 MB<br/>
        • <b>Input Processing:</b> Supports various image formats (JPG, PNG, BMP)<br/>
        • <b>Inference:</b> Can process single images or batches""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("8.3 Model Evaluation Metrics", heading2_style))
    story.append(Paragraph(
        """The project includes comprehensive evaluation including:""",
        body_style
    ))
    story.append(Paragraph(
        """• Classification reports with precision, recall, and F1-score<br/>
        • Confusion matrices for visual analysis<br/>
        • Training curves (loss and accuracy over epochs)<br/>
        • Validation accuracy tracking""",
        body_style
    ))
    
    story.append(PageBreak())
    
    # 9. Project Structure
    story.append(Paragraph("9. Project Structure", heading1_style))
    
    story.append(Paragraph(
        """The project is organized as follows:""",
        body_style
    ))
    
    project_structure = """
FakeImageDetector/
├── ai_generated_detection.ipynb      # EfficientNet-B4 AI detection notebook
├── fake-image-detection.ipynb        # ELA + CNN detection notebook
├── metadata_analysis.ipynb           # EXIF metadata analysis notebook
├── generate_diffusion_images.py      # Script to generate diffusion images
├── datasets/                         # Dataset directories
│   ├── ai_detection/
│   │   ├── real/                     # Real images
│   │   ├── gan/                      # GAN-generated images
│   │   ├── diffusion/                # Diffusion-generated images
│   │   └── test/                     # Test images
│   ├── train/
│   │   ├── real/                     # Real training images
│   │   └── fake/                     # Fake training images
│   └── dataset.csv                   # Dataset metadata
├── models/                           # Trained models
│   ├── ai_generated/
│   │   ├── ai_detector_best.keras
│   │   ├── ai_detector_final.keras
│   │   └── training_history.pkl
│   ├── fake_image_detector_model.keras
│   └── training_history.pkl
├── docs/                             # Documentation
│   ├── model-architecture.jpg
│   └── Deteksi Pemalsuan Gambar dengan ELA dan Deep Learning.pdf
├── AI_DETECTOR_TRAINING_STEPS.md     # Training guide for AI detection
├── AI_DETECTION_DATASET_GUIDE.md     # Dataset preparation guide
├── TRAINING_GUIDE.md                 # Training guide for ELA detection
├── README.md                         # Project readme
└── LICENSE                           # License file
"""
    
    story.append(Preformatted(project_structure, styles['Code'], maxLineLength=100))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("9.1 Key Files", heading2_style))
    story.append(Paragraph(
        """<b>Notebooks:</b><br/>
        • <b>ai_generated_detection.ipynb:</b> Main notebook for AI-generated image detection 
        using EfficientNet-B4<br/>
        • <b>fake-image-detection.ipynb:</b> Notebook for ELA-based tampering detection<br/>
        • <b>metadata_analysis.ipynb:</b> Notebook for EXIF metadata analysis""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Documentation:</b><br/>
        • Comprehensive training guides for both detection methods<br/>
        • Dataset preparation and organization guides<br/>
        • Architecture documentation""",
        body_style
    ))
    
    story.append(PageBreak())
    
    # 10. Conclusion
    story.append(Paragraph("10. Conclusion", heading1_style))
    
    story.append(Paragraph(
        """FakeImageDetector provides a comprehensive solution for detecting fake, tampered, 
        and AI-generated images. The project successfully combines multiple detection techniques:""",
        body_style
    ))
    
    story.append(Paragraph(
        """<b>Key Achievements:</b><br/>
        • Implemented ELA + CNN method achieving 91.83% accuracy<br/>
        • Developed EfficientNet-B4 based AI detection for modern synthetic images<br/>
        • Created comprehensive metadata analysis tools<br/>
        • Provided detailed documentation and training guides""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        """<b>Applications:</b><br/>
        • Journalism and fact-checking<br/>
        • Social media content verification<br/>
        • Digital forensics<br/>
        • Security and authentication systems<br/>
        • Research and education""",
        body_style
    ))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        """<b>Future Enhancements:</b><br/>
        • Integration of multiple detection methods for ensemble predictions<br/>
        • Real-time detection API<br/>
        • Mobile application deployment<br/>
        • Expanded dataset for better generalization<br/>
        • Additional detection methods (PRNU, frequency domain analysis)""",
        body_style
    ))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        """This project demonstrates the effectiveness of combining traditional image forensics 
        techniques with modern deep learning approaches to address the growing challenge of 
        image authenticity verification in the digital age.""",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated successfully: {filename}")
    return filename

if __name__ == "__main__":
    print("Generating FakeImageDetector Report...")
    print("=" * 60)
    
    print("\n1. Creating architecture diagrams...")
    create_architecture_diagram_ela()
    print("   ✓ ELA + CNN architecture diagram created")
    
    create_architecture_diagram_ai()
    print("   ✓ EfficientNet-B4 architecture diagram created")
    
    create_system_overview_diagram()
    print("   ✓ System overview diagram created")
    
    create_data_flow_diagram()
    print("   ✓ Data flow diagram created")
    
    print("\n2. Generating PDF report...")
    pdf_file = create_pdf_report()
    
    print("\n" + "=" * 60)
    print(f"✓ Report generation complete!")
    print(f"✓ Output file: {pdf_file}")
    print(f"✓ Diagrams saved in: report_diagrams/")
    print("=" * 60)

