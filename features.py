# features.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

def compute_magnitude(df):
    """
    Compute magnitude from X, Y, Z accelerometer data.
    Flexible version that checks for different column naming conventions.
    """
    df = df.copy()
    
    # Check for different possible column names
    x_col = None
    y_col = None
    z_col = None
    
    # Look for X column
    possible_x = ['X (m/s^2)', 'acc_x', 'acceleration_x', 'ax', 'Acc_X']
    for col in possible_x:
        if col in df.columns:
            x_col = col
            break
    
    # Look for Y column
    possible_y = ['Y (m/s^2)', 'acc_y', 'acceleration_y', 'ay', 'Acc_Y']
    for col in possible_y:
        if col in df.columns:
            y_col = col
            break
    
    # Look for Z column
    possible_z = ['Z (m/s^2)', 'acc_z', 'acceleration_z', 'az', 'Acc_Z']
    for col in possible_z:
        if col in df.columns:
            z_col = col
            break
    
    # If we found all columns, compute magnitude
    if x_col and y_col and z_col:
        df['magnitude'] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
    elif 'magnitude' in df.columns:
        # Magnitude already computed by csv_loader
        pass
    else:
        # Try to find any acceleration columns
        acc_cols = [col for col in df.columns if any(acc_term in col.lower() 
                     for acc_term in ['acc', 'x', 'y', 'z', 'm/s'])]
        if len(acc_cols) >= 3:
            # Use first 3 acceleration-like columns
            df['magnitude'] = np.sqrt(df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2)
        else:
            raise ValueError(f"Cannot compute magnitude. Missing acceleration columns. Found columns: {list(df.columns)}")
    
    return df


def extract_features(df, segment_duration=2.0):
    """
    Extract features from accelerometer data.
    
    Args:
        df: DataFrame with 'Time (s)' and 'magnitude' columns
        segment_duration: Window size in seconds
    
    Returns:
        DataFrame of extracted features
    """
    features = []
    
    # Check required columns
    required_cols = ['Time (s)', 'magnitude']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available columns: {list(df.columns)}")
    
    start_time = df['Time (s)'].min()
    end_time = df['Time (s)'].max()
    
    for seg_start in np.arange(start_time, end_time, segment_duration):
        seg_end = seg_start + segment_duration
        
        # Extract segment
        seg_df = df[(df['Time (s)'] >= seg_start) & (df['Time (s)'] < seg_end)]
        
        if len(seg_df) < 10:  # Minimum samples for meaningful features
            continue
        
        mag = seg_df['magnitude'].values
        
        # Basic statistics
        mean_mag = np.mean(mag)
        rms_mag = np.sqrt(np.mean(mag**2))
        std_mag = np.std(mag)
        p2p_mag = np.max(mag) - np.min(mag)
        
        # Shape features
        kur_mag = kurtosis(mag) if len(mag) > 3 else 0
        skew_mag = skew(mag) if len(mag) > 3 else 0
        median_mag = np.median(mag)
        q75, q25 = np.percentile(mag, [75, 25])
        iqr_mag = q75 - q25
        
        # Peak count
        peaks, _ = find_peaks(mag, height=np.mean(mag))
        peak_count = len(peaks)
        
        # Collect features
        features.append({
            'segment_start': seg_start,
            'segment_end': seg_end,
            'duration': seg_end - seg_start,
            'sample_count': len(seg_df),
            
            # Motion features
            'mean_magnitude': mean_mag,
            'rms_magnitude': rms_mag,
            'std_magnitude': std_mag,
            'p2p_magnitude': p2p_mag,
            
            # Shape features
            'kurtosis_magnitude': kur_mag,
            'skewness_magnitude': skew_mag,
            'median_magnitude': median_mag,
            'iqr_magnitude': iqr_mag,
            'peak_count': peak_count
        })
    
    return pd.DataFrame(features)