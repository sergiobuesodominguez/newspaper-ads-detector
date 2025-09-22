#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
# Install detectron2 matching the Torch in requirements
pip install 'git+https://github.com/facebookresearch/detectron2.git'
echo ""
echo "✅ Setup complete. To run:"
echo "source .venv/bin/activate && python run.py --input Print_front_pages/ruraljpg --output ads_output_rural --log ads_logsss4.xlsx"
