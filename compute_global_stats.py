"""One-shot script: compute global COORD_MIN/MAX across all CSVs and print."""
import glob
import pandas as pd
import numpy as np

csvs = sorted(glob.glob('data/csv/*.csv'))
global_min = np.array([np.inf, np.inf])
global_max = np.array([-np.inf, -np.inf])
skipped = 0

for path in csvs:
    try:
        df = pd.read_csv(path)
        if not {'x', 'y'}.issubset(df.columns):
            if len(df.columns) >= 2:
                df.columns = ['x', 'y'] + list(df.columns[2:])
            else:
                skipped += 1
                continue
        coords = df[['x', 'y']].values.astype(np.float32)
        global_min = np.minimum(global_min, coords.min(axis=0))
        global_max = np.maximum(global_max, coords.max(axis=0))
    except Exception as e:
        skipped += 1

print(f"Processed {len(csvs) - skipped}/{len(csvs)} CSVs, skipped {skipped}")
print(f"COORD_MIN_X = {global_min[0]}")
print(f"COORD_MIN_Y = {global_min[1]}")
print(f"COORD_MAX_X = {global_max[0]}")
print(f"COORD_MAX_Y = {global_max[1]}")
print(f"Range X = {global_max[0] - global_min[0]}, Range Y = {global_max[1] - global_min[1]}")
