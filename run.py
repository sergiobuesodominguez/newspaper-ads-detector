"""
Newspaper Ad Detector — maintainer notes (written for my future self, ~2–3 years out)

Purpose
-------
This script scans a folder of newspaper front pages (PDFs or images), detects blocks labeled
as "Advertisement" using the Newspaper Navigator model (via LayoutParser + Detectron2),
saves visualizations/crops for each detected ad, and logs coverage statistics to Excel.

What to know before you touch this
----------------------------------
- We only render the FIRST page of a PDF (intentional for front pages).
- Detection is performed with `lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config`
  and a tunable confidence threshold (default 0.7). Push it up to reduce false positives,
  down to catch more borderline cases.
- Output layout: for each input file, we create a sibling folder under `--output` that
  mirrors the relative path. Inside: `detected_ads.jpg` (boxes drawn) + `ad_*.jpg` crops.
- Excel log (`--log`) collects per-file stats and is append-only. If it exists, we append;
  if not, we create it.
- Dependencies: PyMuPDF (fitz), OpenCV, Pillow, pandas, layoutparser (with Detectron2 backend).

Caveats / gotchas
-----------------
- PDFs with odd color spaces or alpha channels are normalized to RGB.
- Some very large PDFs might be slow to rasterize even at 2× zoom.
- We compute area coverage as the union-free sum of ad box areas (i.e., overlaps are
  double-counted if the detector overlaps two boxes). This is acceptable for our use-case,
  but change if you need exact union area.
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
import layoutparser as lp


def load_image(file_path: str):
    """
    Load the image we will run detection on.

    Why it exists:
      - Inputs may be PDFs (front pages) or plain image files (JPG/PNG).
      - For PDFs we only read the *first* page (front page assumption) and rasterize it.
      - We normalize color channels so the detector always sees an RGB-like image.

    Parameters
    ----------
    file_path : str
        Path to a PDF or image file.

    Returns
    -------
    np.ndarray
        An OpenCV-style BGR image array suitable for downstream processing.

    Behavior details / design choices
    ---------------------------------
    - For PDFs:
        * We open with PyMuPDF, render page 0 at 2× zoom for a good balance of speed/detail.
        * We convert RGBA and grayscale into 3-channel RGB, then return BGR for OpenCV.
    - For images:
        * We rely on cv2.imread. If it fails, we raise a helpful FileNotFoundError.
    """
    if file_path.lower().endswith(".pdf"):
        # Render first page of the PDF at 2× resolution.
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for higher fidelity
        image_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Normalize to 3-channel RGB if needed.
        if pix.n == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

        # OpenCV expects BGR.
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at: {file_path}")
        return image


def build_model(score_thresh: float = 0.7):
    """
    Construct the Newspaper Navigator detection model via LayoutParser.

    Why it exists:
      - Centralizes model configuration and label mapping.
      - Keeps the main flow clean and makes thresholding easy to tweak.

    Parameters
    ----------
    score_thresh : float, default=0.7
        Minimum confidence for detections. Raise to reduce false positives.

    Returns
    -------
    lp.models.Detectron2LayoutModel
        A ready-to-use LayoutParser Detectron2 model.

    Notes
    -----
    - We load a pretrained configuration referenced by an `lp://` URI.
    - Label map is aligned with Newspaper Navigator categories. We primarily
      consume "Advertisement", but others remain mapped for clarity.
    """
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
    """
    Run detection on a single file and write per-file outputs.

    What it does (in order):
      1) Loads the first page/image into memory.
      2) Runs the detector and filters to "Advertisement" blocks.
      3) Computes ad pixel coverage and % of the page area.
      4) Creates an output folder that mirrors the input's relative path.
      5) Saves a visualization with boxes and individual ad crops.
      6) Returns a dict of per-file stats (for Excel logging).

    Parameters
    ----------
    model : lp.models.Detectron2LayoutModel
        The LayoutParser model created by `build_model`.
    file_path : str
        Absolute path to the input PDF/JPG/PNG.
    input_root : str
        Root of the input tree (used to compute relative paths for neat output structure).
    output_root : str
        Root folder where we write visualizations/crops and the Excel log.

    Returns
    -------
    dict
        Keys: file, total_pixels, ad_pixels, ad_percentage

    Implementation notes
    --------------------
    - We convert between RGB (for drawing/cropping) and BGR (for cv2.imwrite) as needed.
    - Ad coverage is a simple sum of bounding box areas; we don’t de-overlap.
    - If anything goes wrong for a file, the caller is expected to catch/log and continue.
    """
    # Load the image (BGR) and convert to RGB for the detector/visualization utils.
    image_bgr = load_image(file_path)
    image_rgb = image_bgr[..., ::-1]

    # Run detection and filter to just "Advertisement" blocks.
    layout = model.detect(image_rgb)
    ad_blocks = lp.Layout([b for b in layout if b.type == "Advertisement"])
    print(f"🔎 {os.path.basename(file_path)} → {len(ad_blocks)} ad blocks")

    # Compute total ad area vs page area.
    ad_area = sum(
        (int(b.block.x_2) - int(b.block.x_1)) * (int(b.block.y_2) - int(b.block.y_1))
        for b in ad_blocks
    )
    h, w = image_rgb.shape[:2]
    total_area = h * w
    ad_percentage = (ad_area / total_area * 100) if total_area > 0 else 0

    # Build an output directory that mirrors the input's relative path.
    rel_path = os.path.relpath(file_path, input_root)
    file_stem, _ = os.path.splitext(rel_path)
    out_dir = os.path.join(output_root, file_stem)
    os.makedirs(out_dir, exist_ok=True)

    # Visualization: draw boxes over ads and save.
    viz_pil = lp.draw_box(image_rgb, ad_blocks, box_width=3)
    viz_bgr = cv2.cvtColor(np.array(viz_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "detected_ads.jpg"), viz_bgr)

    # Save per-ad crops to help with manual QA or downstream OCR.
    for i, block in enumerate(ad_blocks, start=1):
        crop = block.crop_image(image_rgb)
        crop_pil = Image.fromarray(crop)
        crop_pil.save(os.path.join(out_dir, f"ad_{i}.jpg"))

    return {
        "file": rel_path,                          # relative path for stable logging across machines
        "total_pixels": total_area,                # page area (pixels)
        "ad_pixels": ad_area,                      # raw sum of ad box areas
        "ad_percentage": round(ad_percentage, 2),  # % of page claimed by ads (approx.)
    }


def main():
    """
    CLI entry point.

    Responsibilities:
      - Parse command-line args.
      - Initialize output directory and the detection model.
      - Walk the input tree, process each supported file, and accumulate stats.
      - Append results to an Excel log (create if missing).

    Arguments
    ---------
    --input (required) :
        Root folder containing PDFs/JPGs/PNGs. We recurse through subfolders.
    --output (required):
        Root folder where per-file outputs and the Excel log are written.
    --log :
        Excel filename under `--output`. Defaults to 'ads_log.xlsx'.
    --score :
        Detection confidence threshold (0.0–1.0). Default 0.7.
    --extensions :
        File extensions to include. Defaults to .pdf, .jpg, .jpeg, .png.

    Exit behavior
    -------------
    - Prints a summary line with the location of the Excel log.
    - Non-fatal errors per-file are reported to stdout; we skip and continue.
    """
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

    # Build the detector once and reuse across files.
    model = build_model(score_thresh=args.score)

    # Walk the input tree and process eligible files.
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
                    # Non-fatal: log and advance. Keeps long runs from dying on one bad file.
                    print(f"⚠️ Skipping {file_path}: {e}")

    # Append to existing log if present; otherwise create a new one.
    df_new = pd.DataFrame(results)
    if os.path.exists(log_path):
        df_existing = pd.read_excel(log_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(log_path, index=False)
    print(f"\n✅ All stats logged to {log_path}")


if __name__ == "__main__":
    # Standard Python entry guard for CLI usage.
    main()
