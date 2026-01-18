from PIL import Image
import numpy as np

def clean_checkerboard(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)

        # Silhouette is likely black/dark. Checkerboard is white/grey.
        # We define a threshold. Anything bright is transparency.
        # R, G, B all > 200 implies white/light grey.
        
        red, green, blue, alpha = data.T

        # Define white/grey areas (approx checkerboard colors)
        # Usually checkerboard is #FFFFFF and #CCCCCC (204)
        white_areas = (red > 180) & (green > 180) & (blue > 180)
        
        # Set alpha to 0 for these areas
        data[..., 3][white_areas.T] = 0

        # Save
        new_img = Image.fromarray(data)
        new_img.save(output_path)
        print(f"Saved cleaned image to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_checkerboard("frontend/public/basketball.png", "frontend/public/basketball_clean.png")
