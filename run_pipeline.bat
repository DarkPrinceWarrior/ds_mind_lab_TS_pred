@echo off
REM WLPR Pipeline Execution Script
REM Usage: run_pipeline.bat [options]

echo ================================================================================
echo WLPR Forecasting Pipeline
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Default parameters
set DATA_PATH=MODEL_22.09.25.csv
set COORDS_PATH=coords.txt
set DISTANCES_PATH=well_distances.xlsx
set OUTPUT_DIR=artifacts
set LOG_LEVEL=INFO

REM Parse arguments (simple version)
if "%1"=="--help" (
    echo Usage: run_pipeline.bat [--with-mlflow] [--debug]
    echo.
    echo Options:
    echo   --with-mlflow    Enable MLflow experiment tracking
    echo   --debug          Enable debug logging
    echo   --skip-cache     Disable caching
    echo   --help           Show this help message
    exit /b 0
)

REM Build command
set CMD=python src/wlpr_pipeline.py --data-path %DATA_PATH% --coords-path %COORDS_PATH% --distances-path %DISTANCES_PATH% --output-dir %OUTPUT_DIR% --log-level %LOG_LEVEL%

REM Add optional flags
if "%1"=="--with-mlflow" set CMD=%CMD% --enable-mlflow
if "%1"=="--debug" set CMD=%CMD% --log-level DEBUG
if "%2"=="--with-mlflow" set CMD=%CMD% --enable-mlflow
if "%2"=="--debug" set CMD=%CMD% --log-level DEBUG
if "%1"=="--skip-cache" set CMD=%CMD% --disable-cache
if "%2"=="--skip-cache" set CMD=%CMD% --disable-cache

echo Starting pipeline with command:
echo %CMD%
echo.

REM Run pipeline
%CMD%

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo Pipeline execution FAILED
    echo Check logs in %OUTPUT_DIR%\logs\
    echo ================================================================================
    exit /b 1
) else (
    echo.
    echo ================================================================================
    echo Pipeline execution COMPLETED
    echo Results saved to: %OUTPUT_DIR%
    echo ================================================================================
)

REM Open results folder
if exist %OUTPUT_DIR% (
    echo.
    echo Opening results folder...
    start "" "%OUTPUT_DIR%"
)

exit /b 0
