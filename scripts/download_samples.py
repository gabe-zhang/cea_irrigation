"""Download real online plant and crop images from public datasets for validation."""

import urllib.request
from pathlib import Path
import cv2
import numpy as np

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Real online agricultural plant photos from open PlantVillage dataset
URLS = {
    "real_tomato_healthy.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Tomato___healthy/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL%20Leaf%20259.1.JPG"
    ),
    "real_tomato_chlorosis.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus/00139ae8-d881-4edb-925f-46584b0bd68c___YLCV_NREC%202944.JPG"
    ),
    "real_tomato_early_blight.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Tomato___Early_blight/0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B%208389.JPG"
    ),
    "real_bell_pepper_healthy.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Pepper%2C_bell___healthy/00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL%207886.JPG"
    ),
    "real_tomato_chlorosis_spidermite.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Tomato___Spider_mites%20Two-spotted_spider_mite/002835d1-c18e-4471-aa6e-8d8c29585e9b___Com.G_SpM_FL%208584.JPG"
    ),
    "real_potato_late_blight.jpg": (
        "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
        "Potato___Late_blight/0051e5e8-d1c4-4a84-bf3a-a426cdad6285___RS_LB%204640.JPG"
    ),
    # Real whole potted plant images for object detection & canopy segmentation
    "potted_plant_monstera.jpg": (
        "https://images.unsplash.com/photo-1485955900006-10f4d324d411?w=640&q=80"
    ),
    "potted_plant_succulent.jpg": (
        "https://images.unsplash.com/photo-1509423350716-97f9360b4e09?w=640&q=80"
    ),
    "potted_plant_indoor.jpg": (
        "https://images.unsplash.com/photo-1545241047-6083a3684587?w=640&q=80"
    ),
    "potted_plant_greenhouse.jpg": (
        "https://images.unsplash.com/photo-1459411552884-841db9b3cc2a?w=640&q=80"
    ),
}


def download_real_samples() -> list[Path]:
    """Download authentic real-world plant dataset images. No synthetic images used."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    downloaded_paths = []

    for filename, url in URLS.items():
        target = SAMPLE_DIR / filename
        if target.exists() and target.stat().st_size > 2000:
            downloaded_paths.append(target)
            continue

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 2000:
                    with open(target, "wb") as f:
                        f.write(data)
                    print(f"[Downloaded Real Photo] {filename} ({len(data)} bytes)")
                    downloaded_paths.append(target)
                    continue
        except Exception as e:
            print(f"[Error] Failed to download real photo {url}: {e}")

    return downloaded_paths


if __name__ == "__main__":
    download_real_samples()
