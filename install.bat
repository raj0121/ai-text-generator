@echo off
echo Installing AI Text Generator Dependencies...
echo.

:: Check Python
python --version
if errorlevel 1 (
    echo Python not found! Install Python 3.8+ from python.org
    pause
    exit
)

echo Step 1: Installing basic dependencies...
pip install --upgrade pip
pip install streamlit==1.28.0 numpy==1.24.0

echo.
echo Step 2: Installing Transformers (this may take a few minutes)...
pip install transformers==4.35.0 sentencepiece==0.1.99 protobuf==3.20.3

echo.
echo Step 3: Installing PyTorch CPU version...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installation complete!
echo.
echo To run the app, execute:
echo   streamlit run app.py
echo.
pause