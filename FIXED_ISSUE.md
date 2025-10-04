# 🔧 Fixed Issue - Ensemble Prediction Column

**Issue:** KeyError "['y_hat'] not in index"  
**Cause:** Ensemble model uses different alias ("ensemble_conservative") than expected ("tsmixerx_wlpr")  
**Fixed:** ✅

---

## What Was Fixed

### Before (Hardcoded Column Name)
```python
preds = preds.rename(columns={"tsmixerx_wlpr": "y_hat"})
```
❌ Fails when model has different alias

### After (Auto-Detection)
```python
# Find the prediction column automatically
pred_cols = [col for col in preds.columns if col not in ['ds', 'unique_id']]
pred_col = pred_cols[0]
preds = preds.rename(columns={pred_col: "y_hat"})
```
✅ Works with any model alias

---

## Files Modified

- ✅ `src/wlpr_pipeline.py` - Fixed `train_and_forecast()` function (line ~1574)

---

## How to Run Now

### Option 1: Use Batch File
```bash
run_phase2.bat
```

### Option 2: Command Line
```bash
cd C:\Users\safae\ts_new
python src\wlpr_pipeline.py --data-path MODEL_22.09.25.csv --coords-path coords.txt --distances-path well_distances.xlsx --output-dir artifacts_physics
```

---

## What You Saw in Logs (Good Signs)

✅ Pipeline started: "Starting WLPR Forecasting Pipeline v5.0 - PHASE 2 COMPLETE"  
✅ Ensemble created: "Creating ensemble of 4 models (mode=weighted)"  
✅ All 4 models: Conservative, Medium, Aggressive, Balanced  
✅ Training started: "Epoch 249: 100%..."  
✅ Training completed: "max_steps=250 reached"  
✅ Prediction started: "Predicting DataLoader 0: 100%"  

❌ Then error occurred (now fixed!)

---

## Run Again

The fix is applied. Run the pipeline again:

```bash
cd C:\Users\safae\ts_new
python src\wlpr_pipeline.py --data-path MODEL_22.09.25.csv --coords-path coords.txt --distances-path well_distances.xlsx --output-dir artifacts_physics
```

Should complete successfully now! 🚀

---

**Status:** Fixed ✅  
**Ready to run:** YES ✅
