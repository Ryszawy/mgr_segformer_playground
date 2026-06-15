@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I "%~1"=="-h" goto :help
if /I "%~1"=="--help" goto :help

if "%~1"=="" (
  for %%I in ("%~dp0..") do set "ROOT=%%~fI"
) else (
  for %%I in ("%~1") do set "ROOT=%%~fI"
)

mkdir "%ROOT%\data" 2>nul
mkdir "%ROOT%\data\cityscapes" 2>nul
mkdir "%ROOT%\data\cityscapes\leftImg8bit\train" 2>nul
mkdir "%ROOT%\data\cityscapes\leftImg8bit\val" 2>nul
mkdir "%ROOT%\data\cityscapes\leftImg8bit\test" 2>nul
mkdir "%ROOT%\data\cityscapes\gtFine\train" 2>nul
mkdir "%ROOT%\data\cityscapes\gtFine\val" 2>nul
mkdir "%ROOT%\data\cityscapes\gtFine\test" 2>nul
mkdir "%ROOT%\data\pretrained_models" 2>nul
mkdir "%ROOT%\data\trained_models" 2>nul
mkdir "%ROOT%\results" 2>nul
mkdir "%ROOT%\results\boundary" 2>nul
mkdir "%ROOT%\results\boundary\viz" 2>nul
mkdir "%ROOT%\results\clean_vis" 2>nul
mkdir "%ROOT%\results\dataset_browse" 2>nul
mkdir "%ROOT%\results\learning_curves" 2>nul
mkdir "%ROOT%\results\plots" 2>nul
mkdir "%ROOT%\results\cityscapes" 2>nul
mkdir "%ROOT%\results\cityscapes\1024x1024" 2>nul
mkdir "%ROOT%\results\cityscapes\1024x1024\vis" 2>nul
mkdir "%ROOT%\work_dirs" 2>nul
mkdir "%ROOT%\runpod_cmd" 2>nul

echo Workspace tree ready at: %ROOT%
echo Created or verified:
echo   data\cityscapes\leftImg8bit\train
echo   data\cityscapes\leftImg8bit\val
echo   data\cityscapes\leftImg8bit\test
echo   data\cityscapes\gtFine\train
echo   data\cityscapes\gtFine\val
echo   data\cityscapes\gtFine\test
echo   data\pretrained_models
echo   data\trained_models
echo   results\boundary\viz
echo   results\clean_vis
echo   results\dataset_browse
echo   results\learning_curves
echo   results\plots
echo   results\cityscapes\1024x1024\vis
echo   work_dirs
echo   runpod_cmd
exit /b 0

:help
echo Usage:
echo   %~nx0 [root_dir]
echo.
echo If root_dir is omitted, the script uses the repository root based on the
echo location of this script.
exit /b 0

