"""
Convert HEIC images to JPG format for model prediction.
"""
from pathlib import Path
import subprocess
import sys

def convert_heic_to_jpg(input_dir: str, output_dir: str, max_images: int = None):
    """
    Convert HEIC images to JPG using macOS's sips command.
    
    Args:
        input_dir: Directory containing HEIC images
        output_dir: Directory to save JPG images
        max_images: Maximum number of images to convert (None = all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    heic_files = list(input_path.glob('*.heic'))
    
    if not heic_files:
        print(f"No HEIC files found in {input_dir}")
        return
    
    print(f"Found {len(heic_files)} HEIC files")
    
    if max_images:
        heic_files = heic_files[:max_images]
        print(f"Converting first {max_images} images...")
    else:
        print(f"Converting all {len(heic_files)} images...")
    
    converted = 0
    failed = 0
    
    for heic_file in heic_files:
        jpg_file = output_path / f"{heic_file.stem}.jpg"
        
        try:
            # Use macOS sips command to convert
            result = subprocess.run(
                ['sips', '-s', 'format', 'jpeg', str(heic_file), '--out', str(jpg_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                converted += 1
                if converted <= 5 or converted % 10 == 0:
                    print(f"  ✓ Converted: {heic_file.name} -> {jpg_file.name}")
            else:
                failed += 1
                print(f"  ✗ Failed: {heic_file.name}")
                print(f"    Error: {result.stderr}")
                
        except Exception as e:
            failed += 1
            print(f"  ✗ Error converting {heic_file.name}: {e}")
    
    print(f"\nConversion complete!")
    print(f"  Successful: {converted}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_path}")
    
    return output_path

if __name__ == "__main__":
    input_dir = "data/custom/ecoglasses_bandeja"
    output_dir = "data/custom/ecoglasses_bandeja_jpg"
    
    # Convert all images
    convert_heic_to_jpg(input_dir, output_dir)
