@echo off
setlocal ENABLEDELAYEDEXPANSION
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
echo.
echo ✅ Setup complete. To run:
echo .venv\Scripts\python.exe run.py --input Print_front_pages\ruraljpg --output ads_output_rural --log ads_logsss4.xlsx
