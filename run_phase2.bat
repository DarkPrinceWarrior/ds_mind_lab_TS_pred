@echo off
echo ========================================
echo Running WLPR Pipeline v5.0 - Phase 2
echo ========================================
cd /d C:\Users\safae\ts_new
python src\wlpr_pipeline.py --data-path MODEL_22.09.25.csv --coords-path coords.txt --distances-path well_distances.xlsx --output-dir artifacts_physics
echo.
echo ========================================
echo Pipeline completed!
echo Check results in: artifacts_physics\
echo ========================================
pause
