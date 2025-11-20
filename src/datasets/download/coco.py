"""
COCO Dataset Download Utilities

Functions for downloading COCO128 and full COCO datasets.
"""

from pathlib import Path
from .utils import download_file, extract_zip, create_train_val_split


def download_coco128(data_dir="data"):
    """
    Download COCO128 dataset (128 images from COCO).
    
    This is a tiny subset perfect for quick testing and validation.
    
    Returns:
        Path to the dataset directory, or None if failed
    """
    print("=" * 70)
    print("Downloading COCO128 Dataset")
    print("=" * 70)
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO128 download URL (from Ultralytics)
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = data_dir / "coco128.zip"
    
    # Download
    if not zip_path.exists():
        download_file(url, zip_path)
    else:
        print(f"✓ {zip_path} already exists, skipping download")
    
    # Extract
    extract_to = data_dir / "coco128"
    if not extract_to.exists():
        extract_zip(zip_path, data_dir)
    else:
        print(f"✓ {extract_to} already exists, skipping extraction")
    
    # Verify structure
    coco128_dir = data_dir / "coco128"
    images_dir = coco128_dir / "images" / "train2017"
    labels_dir = coco128_dir / "labels" / "train2017"
    
    if images_dir.exists() and labels_dir.exists():
        num_images = len(list(images_dir.glob("*.jpg")))
        num_labels = len(list(labels_dir.glob("*.txt")))
        print(f"\n✓ COCO128 dataset ready!")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")
        print(f"  Location: {coco128_dir}")
        
        # Create train/val split (90/10)
        create_train_val_split(images_dir, labels_dir, coco128_dir, split_ratio=0.9)
        
        return str(coco128_dir)
    else:
        print("✗ Error: Dataset structure not as expected")
        return None


def download_roboflow_dataset(workspace, project, version, api_key, data_dir="data"):
    """
    Download dataset from Roboflow.
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        api_key: Roboflow API key
        data_dir: Directory to save dataset
        
    Returns:
        Path to the dataset directory, or None if failed
    """
    print("=" * 70)
    print(f"Downloading Roboflow Dataset: {workspace}/{project}/v{version}")
    print("=" * 70)
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("✗ Error: roboflow package not installed")
        print("  Install with: pip install roboflow")
        return None
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=data_dir)
    
    print(f"\n✓ Dataset downloaded to: {dataset.location}")
    return dataset.location

def convert_coco_json_to_yolo(json_file, output_dir, image_dir):
    """
    Convert COCO JSON annotations to YOLO format txt files.
    """
    import json
    from tqdm import tqdm
    
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map category IDs to contiguous indices (0-79)
    # COCO 2017 has 80 classes but IDs go up to 90
    categories = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {cat['id']: i for i, cat in enumerate(categories)}
    
    print(f"Found {len(categories)} categories")
    
    # Create image dict for quick lookup
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    img_anns = {}
    for ann in tqdm(data['annotations'], desc="Processing annotations"):
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    print(f"Converting annotations for {len(img_anns)} images...")
    
    for img_id, anns in tqdm(img_anns.items(), desc="Writing labels"):
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        img_w = img_info['width']
        img_h = img_info['height']
        file_name = img_info['file_name']
        
        # Check if image exists
        if not (Path(image_dir) / file_name).exists():
            continue
            
        txt_name = Path(file_name).stem + ".txt"
        txt_path = output_dir / txt_name
        
        with open(txt_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in cat_id_to_idx:
                    continue
                
                cls_idx = cat_id_to_idx[cat_id]
                bbox = ann['bbox']  # [x, y, w, h]
                
                # Normalize
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                width = bbox[2] / img_w
                height = bbox[3] / img_h
                
                # Clip to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{cls_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def download_coco_full(data_dir="data"):
    """
    Download full COCO 2017 dataset.
    """
    print("=" * 70)
    print("Downloading Full COCO 2017 Dataset")
    print("=" * 70)
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs
    urls = {
        "train_img": "http://images.cocodataset.org/zips/train2017.zip",
        "val_img": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    # Download
    for name, url in urls.items():
        zip_path = data_dir / f"{name}.zip"
        if not zip_path.exists():
            print(f"\\nDownloading {name}...")
            download_file(url, zip_path)
        else:
            print(f"✓ {name} already downloaded")
            
    # Extract
    print("\\nExtracting files...")
    extract_zip(data_dir / "train_img.zip", data_dir)
    extract_zip(data_dir / "val_img.zip", data_dir)
    extract_zip(data_dir / "annotations.zip", data_dir)
    
    # Convert labels
    print("\\nConverting labels to YOLO format...")
    
    # Train
    train_img_dir = data_dir / "train2017"
    train_lbl_dir = data_dir / "train2017_labels" # Temporary
    convert_coco_json_to_yolo(
        data_dir / "annotations" / "instances_train2017.json",
        train_lbl_dir,
        train_img_dir
    )
    
    # Val
    val_img_dir = data_dir / "val2017"
    val_lbl_dir = data_dir / "val2017_labels" # Temporary
    convert_coco_json_to_yolo(
        data_dir / "annotations" / "instances_val2017.json",
        val_lbl_dir,
        val_img_dir
    )
    
    # Organize into final structure
    # data/coco/train/images
    # data/coco/train/labels
    # data/coco/val/images
    # data/coco/val/labels
    
    final_dir = data_dir / "coco"
    final_dir.mkdir(exist_ok=True)
    
    import shutil
    
    print("\\nOrganizing dataset...")
    
    # Move Train
    (final_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (final_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    
    print("Moving train images...")
    for f in train_img_dir.iterdir():
        shutil.move(str(f), str(final_dir / "train" / "images" / f.name))
    train_img_dir.rmdir()
        
    print("Moving train labels...")
    for f in train_lbl_dir.iterdir():
        shutil.move(str(f), str(final_dir / "train" / "labels" / f.name))
    train_lbl_dir.rmdir()
    
    # Move Val
    (final_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (final_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    print("Moving val images...")
    for f in val_img_dir.iterdir():
        shutil.move(str(f), str(final_dir / "val" / "images" / f.name))
    val_img_dir.rmdir()
        
    print("Moving val labels...")
    for f in val_lbl_dir.iterdir():
        shutil.move(str(f), str(final_dir / "val" / "labels" / f.name))
    val_lbl_dir.rmdir()
    
    print(f"\\n✓ Full COCO dataset ready at {final_dir}")
    return str(final_dir)
