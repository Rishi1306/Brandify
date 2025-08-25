import streamlit as st
import cv2
import os
import re
import csv
import numpy as np
import zipfile
import io
from PIL import Image, ImageDraw, ImageFont
import tempfile
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Brand Asset Processor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def extract_design_number(filename):
    """Extract design number from filename"""
    match = re.search(r"(R[-_]?\d+[A-Z]?)", filename)
    return match.group(1) if match else "Unknown"

def load_size_dict_from_csv(csv_content):
    """Load design-size mapping from CSV content"""
    size_dict = {}
    if csv_content:
        csv_string = csv_content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_string))
        for row in csv_reader:
            design_key = row["Design"].replace("_", "-").upper()
            size_dict[design_key] = row["Size"]
    return size_dict

def process_single_image(img_array, filename, logo_array, size_dict, settings):
    """Process a single image with logo, text, and watermark"""
    img = img_array.copy()
    img_h, img_w, _ = img.shape
    
    # Get design information
    design_number = extract_design_number(filename)
    normalized_design = design_number.replace("_", "-").upper()
    size_raw = size_dict.get(normalized_design, "N/A")
    size_text = f"Size: {size_raw}" if size_raw != "N/A" else "Size: N/A"
    
    # Dynamic sizing
    font_size = max(50, int(min(img_w, img_h) / 20))
    letter_spacing = max(2, int(font_size * 0.1))
    shadow_offset = max(2, int(font_size * 0.05))
    
    # Logo sizing and positioning
    logo_scale_factor = min(img_w, img_h) / 1250
    logo_width = int(350 * logo_scale_factor)
    logo_height = int(140 * logo_scale_factor)
    logo_resized = cv2.resize(logo_array, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
    
    x_offset = 30
    y_offset = img_h - logo_height - 20
    
    # Convert to PIL for text rendering
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Create font (using default if custom font not available)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw design number
    text_x = (x_offset + logo_width // 2) - 60
    text_y = y_offset - (font_size + 80)
    current_x = text_x - (len(design_number) * (font_size // 2)) // 2
    
    for char in design_number:
        draw.text((current_x + shadow_offset, text_y + shadow_offset), char, font=font, fill=(0, 0, 0))
        draw.text((current_x, text_y), char, font=font, fill=(255, 255, 255))
        current_x += font_size // 2 + letter_spacing
    
    # Draw size text
    size_x = x_offset
    size_y = text_y - font_size * 2 - 60
    size_text_width = draw.textlength(size_text, font=font) if hasattr(draw, 'textlength') else len(size_text) * font_size // 2
    size_x = x_offset + (logo_width - size_text_width) // 2
    
    draw.text((size_x + shadow_offset, size_y + shadow_offset), size_text, font=font, fill=(0, 0, 0))
    draw.text((size_x, size_y), size_text, font=font, fill=(255, 255, 255))
    
    # Add watermark if enabled
    if settings['add_watermark']:
        watermark_font_size = max(40, int(min(img_w, img_h) / 18))
        try:
            watermark_font = ImageFont.truetype("arial.ttf", watermark_font_size)
        except:
            watermark_font = ImageFont.load_default()
        
        watermark_overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(watermark_overlay)
        
        text_width = overlay_draw.textlength(settings['watermark_text'], font=watermark_font) if hasattr(overlay_draw, 'textlength') else len(settings['watermark_text']) * watermark_font_size // 2
        center_x = (img_w - text_width) // 2
        center_y = (img_h - watermark_font_size) // 2
        
        overlay_draw.text((center_x, center_y), settings['watermark_text'], 
                         font=watermark_font, fill=(255, 255, 255, settings['watermark_opacity']))
        
        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), watermark_overlay).convert("RGB")
    
    # # Convert back to OpenCV and add logo
    # img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # img[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width] = logo_resized
    
    # return img, design_number, size_raw

        # Paste logo using PIL to preserve transparency
    logo_pil = Image.fromarray(cv2.cvtColor(logo_resized, cv2.COLOR_BGR2RGB)).convert("RGBA")
    img_pil = img_pil.convert("RGBA")
    img_pil.paste(logo_pil, (x_offset, y_offset), logo_pil)

    # Convert final back to OpenCV
    img = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    return img, design_number, size_raw


def convert_to_webp_bytes(img_array, quality=90):
    """Convert image array to WebP bytes"""
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    # Resize if needed
    max_size = (2500, 2500)
    img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    webp_bytes = io.BytesIO()
    img_pil.save(webp_bytes, format='WebP', quality=quality, method=6)
    webp_bytes.seek(0)
    
    return webp_bytes.getvalue()

def create_download_zip(processed_images):
    """Create a ZIP file with all processed images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_data in processed_images:
            # Add original processed image
            img_bytes = io.BytesIO()
            cv2.imwrite('.temp.jpg', img_data['processed_img'])
            with open('.temp.jpg', 'rb') as f:
                img_bytes.write(f.read())
            os.remove('.temp.jpg')
            zip_file.writestr(f"processed/{img_data['filename']}", img_bytes.getvalue())
            
            # Add WebP version
            webp_filename = f"webp/ahana-creations-lace-{os.path.splitext(img_data['filename'])[0]}.webp"
            zip_file.writestr(webp_filename, img_data['webp_bytes'])
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Main App Interface
st.markdown('<div class="main-header"><h1>🎨 Brand Asset Processor</h1><p>Professional image processing with logo, watermark, and WebP conversion</p></div>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Processing Settings")
    
    # Logo upload
    st.subheader("📷 Logo Upload")
    logo_file = st.file_uploader("Upload your logo", type=['png', 'jpg', 'jpeg'])
    
    # CSV upload
    st.subheader("📋 Size Data (CSV)")
    csv_file = st.file_uploader("Upload design sizes CSV", type=['csv'])
    
    st.subheader("🎯 Watermark Settings")
    add_watermark = st.checkbox("Add Watermark", value=True)
    watermark_text = st.text_input("Watermark Text", value="AHANA CREATIONS")
    watermark_opacity = st.slider("Watermark Opacity", 50, 200, 120)
    
    st.subheader("🌐 WebP Settings")
    webp_quality = st.slider("WebP Quality", 60, 100, 90)
    
    settings = {
        'add_watermark': add_watermark,
        'watermark_text': watermark_text,
        'watermark_opacity': watermark_opacity,
        'webp_quality': webp_quality
    }

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and logo_file:
        st.success(f"✅ {len(uploaded_files)} images ready for processing!")
        
        # Process button
        if st.button("🚀 Process Images", type="primary"):
            if not csv_file:
                st.warning("⚠️ CSV file recommended for size information")
            
            # Load data
            logo_array = cv2.imdecode(np.frombuffer(logo_file.read(), np.uint8), cv2.IMREAD_COLOR)
            size_dict = load_size_dict_from_csv(csv_file.read() if csv_file else None)
            
            # Reset session state
            st.session_state.processed_images = []
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process images
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Read image
                img_array = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                # Process image
                processed_img, design_number, size_info = process_single_image(
                    img_array, uploaded_file.name, logo_array, size_dict, settings
                )
                
                # Convert to WebP
                webp_bytes = convert_to_webp_bytes(processed_img, settings['webp_quality'])
                
                # Store results
                st.session_state.processed_images.append({
                    'filename': uploaded_file.name,
                    'original_img': img_array,
                    'processed_img': processed_img,
                    'webp_bytes': webp_bytes,
                    'design_number': design_number,
                    'size_info': size_info
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            st.session_state.processing_complete = True
            st.rerun()

with col2:
    st.subheader("👀 Preview & Results")
    
    if st.session_state.processed_images:
        # Display processing summary
        st.markdown(f'<div class="success-box">🎉 Successfully processed {len(st.session_state.processed_images)} images!</div>', unsafe_allow_html=True)
        
        # Image preview selector
        if len(st.session_state.processed_images) > 0:
            selected_idx = st.selectbox(
                "Select image to preview:",
                range(len(st.session_state.processed_images)),
                format_func=lambda x: st.session_state.processed_images[x]['filename']
            )
            
            selected_img_data = st.session_state.processed_images[selected_idx]
            
            # Show before/after comparison
            st.subheader("Before vs After")
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.text("Original")
                st.image(cv2.cvtColor(selected_img_data['original_img'], cv2.COLOR_BGR2RGB), 
                        use_container_width=True)
            
            with col_after:
                st.text("Processed")
                st.image(cv2.cvtColor(selected_img_data['processed_img'], cv2.COLOR_BGR2RGB), 
                        use_container_width=True)
            
            # Show metadata
            st.markdown(f"""
            **Design Number:** {selected_img_data['design_number']}  
            **Size Info:** {selected_img_data['size_info']}  
            **WebP Size:** {len(selected_img_data['webp_bytes']) // 1024} KB
            """)

# Download section
if st.session_state.processed_images:
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download individual WebP
        if st.session_state.processed_images:
            selected_for_download = st.selectbox(
                "Download individual WebP:",
                range(len(st.session_state.processed_images)),
                format_func=lambda x: st.session_state.processed_images[x]['filename'],
                key="download_select"
            )
            
            if st.button("Download WebP"):
                selected_data = st.session_state.processed_images[selected_for_download]
                webp_filename = f"ahana-creations-lace-{os.path.splitext(selected_data['filename'])[0]}.webp"
                
                st.download_button(
                    label="📱 Click to Download",
                    data=selected_data['webp_bytes'],
                    file_name=webp_filename,
                    mime="image/webp"
                )
    
    with col2:
        # Download all as ZIP
        if st.button("Create ZIP Package"):
            zip_data = create_download_zip(st.session_state.processed_images)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📦 Download All (ZIP)",
                data=zip_data,
                file_name=f"processed_images_{timestamp}.zip",
                mime="application/zip"
            )
    
    with col3:
        # Processing statistics
        total_size_original = sum([img['original_img'].nbytes for img in st.session_state.processed_images]) // (1024*1024)
        total_size_webp = sum([len(img['webp_bytes']) for img in st.session_state.processed_images]) // 1024
        
        st.metric("Original Size", f"{total_size_original} MB")
        st.metric("WebP Size", f"{total_size_webp} KB")
        st.metric("Compression", f"{((total_size_original*1024 - total_size_webp) / (total_size_original*1024) * 100):.1f}%")

# Features section
st.subheader("✨ Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h4>🎯 Smart Processing</h4>
        <ul>
            <li>Automatic design number extraction</li>
            <li>Dynamic logo and text scaling</li>
            <li>CSV-based size mapping</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>🌐 Web Optimization</h4>
        <ul>
            <li>High-quality WebP conversion</li>
            <li>Configurable compression</li>
            <li>Batch processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>🎨 Brand Consistency</h4>
        <ul>
            <li>Automatic logo placement</li>
            <li>Custom watermarks</li>
            <li>Professional styling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 🚀 Ready to process your brand assets? Upload your images and logo to get started!")
st.markdown("*Built with Streamlit for efficient batch image processing and web optimization.*")