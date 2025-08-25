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
    
    # Create font (try multiple font options)
    font = None
    font_paths = [
        "arial.ttf", 
        "Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "C:/Windows/Fonts/arial.ttf"  # Windows
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    
    # Fallback to default font if none found
    if font is None:
        try:
            font = ImageFont.load_default()
            # Scale up the default font size
            font_size = font_size // 2  # Default font is usually smaller
        except:
            font = ImageFont.load_default()
    
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
    
    # Draw design number with shadow
    if len(design_number) > 1:  # Only draw if we have a valid design number
        # Shadow
        draw.text((current_x + shadow_offset, text_y + shadow_offset), design_number, 
                 font=font, fill=(0, 0, 0, 180))
        # Main text
        draw.text((current_x, text_y), design_number, 
                 font=font, fill=(255, 255, 255, 255))
    
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
    
    # Add watermark if enabled
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
    
    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Add logo (ensure it fits within image bounds)
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
        else:
            # Simple overlay for non-transparent logos
            img[y_offset:y_offset+logo_height, x_offset:x_offset+logo_width] = logo_resized
    
    return img, design_number, size_raw