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
    .debug-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 12px;
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
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

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
    
    # Debug information
    debug_info = {
        'filename': filename,
        'design_number': design_number,
        'normalized_design': normalized_design,
        'size_raw': size_raw,
        'image_dimensions': f"{img_w}x{img_h}",
        'available_designs': list(size_dict.keys()) if size_dict else []
    }
    
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
    
    # Create font (try multiple font options)
    font = None
    font_paths = [
        "arial.ttf", 
        "Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS fallback
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"  # Linux fallback
    ]
    
    font_loaded = "default"
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            font_loaded = font_path
            break
        except (OSError, IOError):
            continue
    
    # Fallback to default font if none found
    if font is None:
        try:
            font = ImageFont.load_default()
            font_size = font_size // 2  # Default font is usually smaller
        except:
            font = ImageFont.load_default()
    
    debug_info['font_loaded'] = font_loaded
    debug_info['font_size'] = font_size
    
    # Draw design number with better positioning and visibility
    text_x = x_offset + logo_width // 2
    text_y = y_offset - (font_size + 80)
    
    # Get text bounding box for better centering
    try:
        bbox = draw.textbbox((0, 0), design_number, font=font)
        text_width = bbox[2] - bbox[0]
    except AttributeError:
        # Fallback for older PIL versions
        text_width = len(design_number) * (font_size // 2)
    
    current_x = text_x - (text_width // 2)
    
    # Draw design number with shadow (only if we have a valid design number)
    if design_number != "Unknown" and len(design_number) > 1:
        # Shadow
        draw.text((current_x + shadow_offset, text_y + shadow_offset), design_number, 
                 font=font, fill=(0, 0, 0, 180))
        # Main text
        draw.text((current_x, text_y), design_number, 
                 font=font, fill=(255, 255, 255, 255))
        debug_info['design_text_drawn'] = True
    else:
        debug_info['design_text_drawn'] = False
    
    # Draw size text
    size_y = text_y - font_size - 20
    
    try:
        bbox = draw.textbbox((0, 0), size_text, font=font)
        size_text_width = bbox[2] - bbox[0]
    except AttributeError:
        size_text_width = len(size_text) * (font_size // 3)
    
    size_x = x_offset + (logo_width - size_text_width) // 2
    
    # Draw size text with shadow
    draw.text((size_x + shadow_offset, size_y + shadow_offset), size_text, 
             font=font, fill=(0, 0, 0, 180))
    draw.text((size_x, size_y), size_text, 
             font=font, fill=(255, 255, 255, 255))
    debug_info['size_text_drawn'] = True
    
    # Add watermark if enabled
    watermark_applied = False
    if settings['add_watermark'] and settings['watermark_text'].strip():
        watermark_font_size = max(40, int(min(img_w, img_h) / 18))
        
        # Try to create watermark font
        watermark_font = None
        for font_path in font_paths:
            try:
                watermark_font = ImageFont.truetype(font_path, watermark_font_size)
                break
            except (OSError, IOError):
                continue
        
        if watermark_font is None:
            try:
                watermark_font = ImageFont.load_default()
                watermark_font_size = watermark_font_size // 2
            except:
                watermark_font = ImageFont.load_default()
        
        # Create watermark overlay
        watermark_overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(watermark_overlay)
        
        # Get watermark text dimensions
        try:
            bbox = overlay_draw.textbbox((0, 0), settings['watermark_text'], font=watermark_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width = len(settings['watermark_text']) * (watermark_font_size // 2)
            text_height = watermark_font_size
        
        # Center the watermark
        center_x = (img_w - text_width) // 2
        center_y = (img_h - text_height) // 2
        
        # Draw watermark with specified opacity
        overlay_draw.text((center_x, center_y), settings['watermark_text'], 
                         font=watermark_font, 
                         fill=(255, 255, 255, settings['watermark_opacity']))
        
        # Composite the watermark onto the image
        img_pil = img_pil.convert("RGBA")
        img_pil = Image.alpha_composite(img_pil, watermark_overlay)
        img_pil = img_pil.convert("RGB")
        watermark_applied = True
    
    debug_info['watermark_applied'] = watermark_applied
    debug_info['watermark_settings'] = {
        'enabled': settings['add_watermark'],
        'text': settings['watermark_text'],
        'opacity': settings['watermark_opacity']
    }
    
    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Add logo (ensure it fits within image bounds)
    logo_applied = False
    if (y_offset + logo_height <= img_h and x_offset + logo_width <= img_w):
        # Handle logo with transparency if it's PNG
        if logo_array.shape[2] == 4:  # Has alpha channel
            # Extract alpha channel
            logo_rgb = logo_array[:, :, :3]
            alpha = logo_array[:, :, 3] / 255.0
            
            # Resize alpha channel too
            alpha_resized = cv2.resize(alpha, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
            logo_rgb_resized = cv2.resize(logo_rgb, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
            
            # Apply alpha blending
            for c in range(3):
                img[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width, c] = (
                    alpha_resized * logo_rgb_resized[:, :, c] + 
                    (1 - alpha_resized) * img[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width, c]
                )
            logo_applied = True
        else:
            # Simple overlay for non-transparent logos
            img[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width] = logo_resized
            logo_applied = True
    
    debug_info['logo_applied'] = logo_applied
    debug_info['logo_dimensions'] = f"{logo_width}x{logo_height}"
    debug_info['logo_position'] = f"({x_offset}, {y_offset})"
    
    return img, design_number, size_raw, debug_info

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
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
    
    # Logo upload
    st.subheader("📷 Logo Upload")
    logo_file = st.file_uploader("Upload your logo", type=['png', 'jpg', 'jpeg'])
    
    # CSV upload
    st.subheader("📋 Size Data (CSV)")
    csv_file = st.file_uploader("Upload design sizes CSV", type=['csv'])
    if csv_file:
        # Show CSV preview
        csv_preview = csv_file.read(200).decode('utf-8')
        csv_file.seek(0)  # Reset file pointer
        st.text("CSV Preview:")
        st.code(csv_preview + "..." if len(csv_preview) == 200 else csv_preview)
    
    st.subheader("🎯 Watermark Settings")
    add_watermark = st.checkbox("Add Watermark", value=True)
    watermark_text = st.text_input("Watermark Text", value="AHANA CREATIONS")
    watermark_opacity = st.slider("Watermark Opacity", 50, 255, 120)
    
    st.subheader("🌐 WebP Settings")
    webp_quality = st.slider("WebP Quality", 60, 100, 90)
    
    # Validate and create settings
    settings = {
        'add_watermark': add_watermark,
        'watermark_text': watermark_text.strip() if watermark_text else "AHANA CREATIONS",
        'watermark_opacity': max(50, min(255, watermark_opacity)),
        'webp_quality': webp_quality
    }
    
    if st.session_state.debug_mode:
        st.subheader("🔧 Current Settings")
        st.json(settings)

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
            try:
                logo_array = cv2.imdecode(np.frombuffer(logo_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                if logo_array is None:
                    st.error("❌ Failed to load logo file")
                    st.stop()
                
                size_dict = load_size_dict_from_csv(csv_file.read() if csv_file else None)
                
                if st.session_state.debug_mode:
                    st.info(f"Logo shape: {logo_array.shape}")
                    st.info(f"Loaded {len(size_dict)} size mappings: {list(size_dict.keys())[:5]}...")
                
            except Exception as e:
                st.error(f"❌ Error loading files: {str(e)}")
                st.stop()
            
            # Reset session state
            st.session_state.processed_images = []
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            debug_container = st.container() if st.session_state.debug_mode else None
            
            # Process images
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Read image
                    img_array = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img_array is None:
                        st.error(f"❌ Failed to load {uploaded_file.name}")
                        continue
                    
                    # Process image
                    processed_img, design_number, size_info, debug_info = process_single_image(
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
                        'size_info': size_info,
                        'debug_info': debug_info
                    })
                    
                    # Show debug info
                    if st.session_state.debug_mode and debug_container:
                        with debug_container:
                            st.markdown(f'<div class="debug-box"><strong>{uploaded_file.name}</strong><br>{debug_info}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue
                
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
            
            # Show debug info if enabled
            if st.session_state.debug_mode:
                st.subheader("🐛 Debug Information")
                st.json(selected_img_data['debug_info'])

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
        if total_size_original > 0:
            compression_ratio = ((total_size_original*1024 - total_size_webp) / (total_size_original*1024) * 100)
            st.metric("Compression", f"{compression_ratio:.1f}%")

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
            <li>Debug mode for troubleshooting</li>
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
            <li>Automatic image resizing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h4>🎨 Brand Consistency</h4>
        <ul>
            <li>Automatic logo placement</li>
            <li>Custom watermarks with opacity</li>
            <li>Professional text styling</li>
            <li>Multiple font support</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Troubleshooting section
if st.session_state.debug_mode:
    st.subheader("🔧 Troubleshooting Tips")
    st.markdown("""
    **Common Issues:**
    - **Text not appearing:** Check if your filenames contain design numbers (e.g., R123, R-456)
    - **Wrong sizes:** Ensure CSV has 'Design' and 'Size' columns with matching design numbers
    - **Watermark not visible:** Adjust opacity settings or check watermark text
    - **Logo not showing:** Verify logo file format and try PNG with transparency
    
    **CSV Format Example:**
    ```
    Design,Size
    R123,Medium
    R-456,Large
    R_789A,Small
    ```
    """)

# Footer
st.markdown("---")
st.markdown("### 🚀 Ready to process your brand assets? Upload your images and logo to get started!")
st.markdown("*Built with Streamlit for efficient batch image processing and web optimization.*")