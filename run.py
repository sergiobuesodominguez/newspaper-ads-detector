import argparse
import os
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import layoutparser as lp

def load_image(file_path: str):
    """Load first page of PDF as image, or read image file."""
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
        image_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at: {file_path}")
        return image

def build_model(score_thresh: float = 0.7):
    """Load the Newspaper Navigator Detectron2 model via LayoutParser."""
    model = lp.Detectron2LayoutModel(
        'lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        label_map={
            0: "Photograph",
            1: "Illustration",
            2: "Map",
            3: "Comics/Cartoon",
            4: "Editorial Cartoon",
            5: "Headline",
            6: "Advertisement",
        },
    )
    return model

def process_file(model, file_path: str, input_root: str, output_root: str):
    image_bgr = load_image(file_path)
    image_rgb = image_bgr[..., ::-1]

    layout = model.detect(image_rgb)
    ad_blocks = lp.Layout([b for b in layout if b.type == "Advertisement"])
    print(f"🔎 {os.path.basename(file_path)} → {len(ad_blocks)} ad blocks")

    # Stats
    ad_area = sum((int(b.block.x_2) - int(b.block.x_1)) * (int(b.block.y_2) - int(b.block.y_1)) for b in ad_blocks)
    h, w = image_rgb.shape[:2]
    total_area = h * w
    ad_percentage = (ad_area / total_area * 100) if total_area > 0 else 0

    # Output dirs
    rel_path = os.path.relpath(file_path, input_root)
    file_stem, _ = os.path.splitext(rel_path)
    out_dir = os.path.join(output_root, file_stem)
    os.makedirs(out_dir, exist_ok=True)

    # Visualization
    viz_pil = lp.draw_box(image_rgb, ad_blocks, box_width=3)
    viz_bgr = cv2.cvtColor(np.array(viz_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "detected_ads.jpg"), viz_bgr)

    # Ad crops
    for i, block in enumerate(ad_blocks, start=1):
        crop = block.crop_image(image_rgb)
        crop_pil = Image.fromarray(crop)
        crop_pil.save(os.path.join(out_dir, f"ad_{i}.jpg"))

    return {
        "file": rel_path,
        "total_pixels": total_area,
        "ad_pixels": ad_area,
        "ad_percentage": round(ad_percentage, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Detect newspaper ads and log coverage.")
    parser.add_argument("--input", required=True, help="Root folder containing PDFs/JPGs/PNGs (recursively)")
    parser.add_argument("--output", required=True, help="Output root folder for crops & visualizations")
    parser.add_argument("--log", default="ads_log.xlsx", help="Excel log filename to create under output root")
    parser.add_argument("--score", type=float, default=0.7, help="Detection confidence threshold (0.0–1.0)")
    parser.add_argument("--extensions", nargs="*", default=[".pdf", ".jpg", ".jpeg", ".png"],
                        help="File extensions to include")

    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, args.log)

    model = build_model(score_thresh=args.score)

    results = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if any(f.lower().endswith(ext) for ext in args.extensions):
                file_path = os.path.join(root, f)
                try:
                    res = process_file(model, file_path, input_root, output_root)
                    if res:
                        results.append(res)
                except Exception as e:
                    print(f"⚠️ Skipping {file_path}: {e}")

    df_new = pd.DataFrame(results)
    if os.path.exists(log_path):
        df_existing = pd.read_excel(log_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(log_path, index=False)
    print(f"\n✅ All stats logged to {log_path}")

if __name__ == "__main__":
    main()
