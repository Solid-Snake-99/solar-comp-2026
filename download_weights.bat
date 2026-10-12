@echo off
title Solar Filament Segmentation - Weights Downloader
cd /d "%~dp0"
echo ===========================================================================
echo Solar Filament Segmentation - Model Checkpoints Downloader
echo ===========================================================================
echo.
python download_weights.py
echo.
pause
