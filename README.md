# Newspaper Ads Detector

Detect advertisement blocks on newspaper front pages (PDF/JPG/PNG) using LayoutParser + Detectron2 (Newspaper Navigator model). Outputs cropped ad images, a visualization with boxes, and an Excel log with per‑page ad coverage.

## 📦 Repo Structure
```
newspaper-ads-detector/
├─ README.md
├─ run.py
├─ requirements.txt
├─ install.sh
├─ install.bat
├─ .gitignore
├─ sample_data/
│  └─ README.txt
└─ outputs/                # created at runtime
```

## 🧑‍💻 Who is this for?
**Non‑coders who just want to run it.** You only need to follow the steps below (copy/paste commands). No code editing required.

## 🚀 Quick Start
1) **Download repo**: Click **Code → Download ZIP**, unzip.
2) **Open a terminal** (PowerShell on Windows, Terminal on macOS/Linux), `cd` into the folder.
3) **Run the installer**:
   - **Windows**: `install.bat`
   - **macOS/Linux**: `bash install.sh`
4) **Put your files** (PDF/JPG/PNG) under a folder, e.g. `Print_front_pages/ruraljpg`.
5) **Run**:
   ```bash
   .venv/bin/python run.py --input Print_front_pages/ruraljpg --output ads_output_rural --log ads_logsss4.xlsx
   ```
   On Windows PowerShell:
   ```powershell
   .venv\Scripts\python.exe run.py --input Print_front_pages/ruraljpg --output ads_output_rural --log ads_logsss4.xlsx
   ```

You’ll find crops & visualizations inside `ads_output_rural/...` and stats in `ads_output_rural/ads_logsss4.xlsx`.

## 📝 Requirements & Notes
- Python **3.10–3.12** recommended.
- Internet access on first run to fetch the pre‑trained model.
- This uses Detectron2 (CPU). Installation is automated by the scripts; if it fails, see **Troubleshooting**.

## ⚙️ Options
- `--score`: change detection strictness (0.0–1.0). Example: `--score 0.6` to detect more, risk more false positives.
- `--extensions`: restrict file types, e.g. `--extensions .pdf .jpg`.
- `--log`: choose another log name, e.g. `--log rural_run_sep22.xlsx`.

## ❓ FAQ
**Q: Do I need a GPU?**  
A: No. It runs on CPU. A GPU speeds it up but isn’t required.

**Q: The install fails at detectron2.**  
A: Try upgrading pip (`python -m pip install --upgrade pip`) and re‑running the installer. If it persists, see Troubleshooting.

**Q: Can I re‑run on the same output folder?**  
A: Yes. The Excel log will append new rows to the same file.

## 🧰 Troubleshooting
- **detectron2 build errors**
  - Make sure Python is 3.10–3.12.
  - On Windows, use the **Developer PowerShell for VS** or ensure Build Tools are installed (MSVC). Alternatively, use **WSL** (Ubuntu) and run `bash install.sh` there.
  - Try installing specific wheel versions compatible with your Torch (check detectron2 README). Example CPU install:
    ```bash
    pip install --upgrade pip
    pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

- **PyMuPDF errors**
  - Ensure pip is up‑to‑date. On Apple Silicon, install Xcode Command Line Tools: `xcode-select --install`.

- **OpenCV errors**
  - If GUI support error appears, you can switch to `opencv-python-headless` in `requirements.txt`.

- **No ads detected but there should be**
  - Lower the threshold: `--score 0.6`.

## 🔒 Licensing & Credits
- Model: **Newspaper Navigator** via LayoutParser (Detectron2). See their licenses.
- This repo is MIT licensed (change as you wish).

---

### Push to GitHub (one‑time)
```bash
git init
git add .
git commit -m "feat: initial release of newspaper-ads-detector"
# Create a new empty repo on GitHub named newspaper-ads-detector, then:
git branch -M main
git remote add origin https://github.com/<your-username>/newspaper-ads-detector.git
git push -u origin main
```
