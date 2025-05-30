from flask import Flask, render_template
import os
import argparse
from pathlib import Path
import mimetypes
import sys
sys.path.append('/openseg_blob/zhaoyaqi/flow_grpo')
from sas_key import add_prefix_suffix

app = Flask(__name__)

def is_image_file(filename):
    """Check if a file is an image"""
    mtype, _ = mimetypes.guess_type(filename)
    return mtype and mtype.startswith('image/')

def generate_gallery_html(folder_path, caption_format="{image_name}", zoom_scale=1.2, img_size=200):
    """Generate image gallery HTML
    
    Args:
        folder_path (str): Path to image folder
        caption_format (str): Caption format using {image_name} as placeholder
        zoom_scale (float): Image zoom scale on hover
        img_size (int): Base size of image container in pixels
    """
    # Get folder name for title
    folder_name = os.path.basename(folder_path)
    
    # Get all images from folder
    folder_path = Path(folder_path)
    images = []
    
    # Get all image files
    for file in folder_path.glob('*'):
        if is_image_file(str(file)):
            # Process path with add_prefix_suffix
            image_name = file.name
            caption = caption_format.format(image_name=image_name)
            images.append({
                'name': image_name,
                'caption': caption,
                'path': add_prefix_suffix(str(file.absolute()))
            })
    
    # Sort by name
    images.sort(key=lambda x: x['name'])
    
    # Ensure output directory exists
    output_dir = Path('/openseg_blob/zhaoyaqi/html/result')
    output_dir.mkdir(exist_ok=True)
    
    # Render HTML in app context
    with app.app_context():
        # Render HTML
        html_content = render_template('gallery.html', 
                                     images=images,
                                     zoom_scale=zoom_scale,
                                     img_size=img_size,
                                     title=folder_name)
        
        name = os.path.basename(folder_path)
        # Save HTML file
        output_path = output_dir / f'{name}.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML file generated: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate image gallery HTML')
    parser.add_argument('folder', help='Path to folder containing images')
    parser.add_argument('--caption-format', default="{image_name}",
                      help='Caption format using {image_name} as placeholder, e.g.: ode_{image_name}')
    parser.add_argument('--zoom-scale', type=float, default=1.2,
                      help='Image zoom scale on hover')
    parser.add_argument('--img-size', type=int, default=200,
                      help='Base size of image container in pixels')
    args = parser.parse_args()
    
    # Set template folder path
    app.template_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    
    generate_gallery_html(args.folder,
                         caption_format=args.caption_format,
                         zoom_scale=args.zoom_scale,
                         img_size=args.img_size) 