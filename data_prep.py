import os
import zipfile
import shutil
import pandas as pd
import sys

def create_directories():
    """Create structural directories for the cleaned dataset."""
    base_dir = "data"
    images_dir = os.path.join(base_dir, "images")
    csv_dir = os.path.join(base_dir, "csv")
    
    for d in [base_dir, images_dir, csv_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    return base_dir, images_dir, csv_dir

def draw_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    """Render terminal progress completion bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total: 
        print("")

def process_data(zip_path="one-stroke-dotted-pulli-kolam.zip"):
    """Extract, validate, and structure the raw zipped Kolam dataset."""
    print("Starting data preparation pipeline...")
    
    if not os.path.exists(zip_path):
        print("Error: Could not find '{}' in the current directory.".format(zip_path))
        return
        
    base_dir, images_dir, csv_dir = create_directories()
    temp_extract_dir = "temp_extract"
    
    print("Extracting '{}' to temporary directory...".format(zip_path))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    images = []
    csvs = []
    
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                images.append(os.path.join(root, file))
            elif ext == '.csv':
                csvs.append(os.path.join(root, file))
                
    def get_core_stem(filepath):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        # If it's something like kolam109-0, we just want 'kolam109' to match the CSV
        if '-' in stem:
            return stem.rsplit('-', 1)[0]
        return stem
        
    csv_map = {get_core_stem(f): f for f in csvs}
    
    # Gather pairs by finding the core CSV for each image
    valid_pairs = []
    for img in images:
        core = get_core_stem(img)
        if core in csv_map:
            valid_pairs.append((img, csv_map[core]))
            
    total_pairs = len(valid_pairs)
    
    print("Found {} paired files in the archive. Starting sanity checks and migration...".format(total_pairs))
    
    if total_pairs == 0:
        print("No paired files found. Please verify the contents of the zip file.")
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    valid_count = 0
    discarded_count = 0
    
    draw_progress_bar(0, total_pairs, prefix='Processing:', suffix='Complete', length=50)
    
    for i, (image_path, csv_path) in enumerate(valid_pairs):
        try:
            # 2. Bounding Box Sanity Check
            df = pd.read_csv(csv_path)
            
            # Discard empty or single-point sequence files
            if len(df) <= 1:
                discarded_count += 1
                draw_progress_bar(i + 1, total_pairs, prefix='Processing:', suffix='Complete', length=50)
                continue
                
            # Normalize headers
            if not {'x', 'y'}.issubset(set(df.columns)):
                if len(df.columns) >= 2:
                    df.columns = ['x', 'y'] + list(df.columns[2:])
                else:
                    discarded_count += 1
                    draw_progress_bar(i + 1, total_pairs, prefix='Processing:', suffix='Complete', length=50)
                    continue
                    
            # Compute true bounding box limits to confirm structural integrity
            x_min, x_max = float(df['x'].min()), float(df['x'].max())
            y_min, y_max = float(df['y'].min()), float(df['y'].max())
            
            # Discard totally collapsed vectors where bounding boxes equal zero dimension
            if x_min == x_max and y_min == y_max:
                discarded_count += 1
                draw_progress_bar(i + 1, total_pairs, prefix='Processing:', suffix='Complete', length=50)
                continue
                
            # 3. Processing: Migrate clean files smoothly to their respective domains
            # We rename the CSV to exactly match the image stem so train.py finds it 1-to-1
            img_stem = os.path.splitext(os.path.basename(image_path))[0]
            dest_image_path = os.path.join(images_dir, os.path.basename(image_path))
            dest_csv_path = os.path.join(csv_dir, img_stem + '.csv')
            
            shutil.copy2(image_path, dest_image_path)
            shutil.copy2(csv_path, dest_csv_path)
            
            valid_count += 1
            
        except Exception as e:
            # Silently swallow corrupted CSV parsing exceptions
            discarded_count += 1
            
        # Terminal execution visualization
        draw_progress_bar(i + 1, total_pairs, prefix='Processing:', suffix='Complete', length=50)
        
    print("\nData preparation complete!")
    print("Total Verified & Clean Training Pairs: {}".format(valid_count))
    print("Discarded Corrupt/Empty Sequences: {}".format(discarded_count))
    
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    print("Done. Cleaned dataset is ready in 'data/csv/' and 'data/images/'.")

if __name__ == "__main__":
    process_data()
