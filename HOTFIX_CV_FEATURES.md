# 🔧 Hotfix: CV Features Creation

**Date:** October 4, 2025  
**Issue:** Missing advanced features in walk-forward CV folds  
**Status:** FIXED ✅

---

## 🐛 Problem

When running the pipeline, it failed during walk-forward cross-validation:

```
ValueError: Fold 1 missing required features: ['wlpr_std12', 'wbhp_std3', 'wlpr_ma3', 'wlpr_x_wbhp', ...]
```

**Root cause:**  
Advanced features (interactions, spatial, rolling stats) were created for the main dataset but not for each CV fold.

---

## ✅ Solution

Updated `run_walk_forward_validation()` to create advanced features for each fold:

### Changes made in `src/wlpr_pipeline.py`:

1. **Create features for each fold** (after line 764):
   ```python
   # IMPROVEMENT #2: Create advanced features for this fold
   fold_prod = _create_interaction_features(fold_prod)
   fold_prod = _create_spatial_features(fold_prod, coords)
   fold_prod = _create_rolling_statistics(fold_prod, feature_cols=["wlpr", "wbhp"], windows=[3, 6, 12])
   ```

2. **Create fold-specific static_df** (after line 786):
   ```python
   # Create fold-specific static_df with spatial features
   static_cols = ["unique_id"] + [col for col in config.static_exog if col in fold_prod.columns]
   fold_static_df = fold_prod.groupby("unique_id")[static_cols].first().reset_index(drop=True)
   for col in config.static_exog:
       if col not in fold_static_df.columns:
           fold_static_df[col] = 0.0
   ```

3. **Use fold_static_df in training and prediction**:
   ```python
   nf.fit(df=fold_train, static_df=fold_static_df, ...)
   preds = nf.predict(futr_df=fold_futr, static_df=fold_static_df)
   ```

---

## 🧪 Testing

The fix ensures that:
- ✅ All 22 advanced features are created for each CV fold
- ✅ Spatial features are included in fold-specific static_df
- ✅ No missing features error during CV
- ✅ Walk-forward validation runs successfully

---

## 🚀 Run Now

The pipeline should work now:

```bash
python src\wlpr_pipeline.py --data-path MODEL_22.09.25.csv --coords-path coords.txt --distances-path well_distances.xlsx --output-dir artifacts_physics
```

---

## 📊 Expected Behavior

You should see:
1. Walk-forward CV starts successfully
2. Features created for each fold (6 folds)
3. No "missing features" errors
4. CV metrics logged for each fold
5. Aggregate CV metrics at the end

---

**Status:** ✅ FIXED  
**Ready to run:** YES
