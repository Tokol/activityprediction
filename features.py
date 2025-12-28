import pandas as pd
import numpy as np

def compute_magnitude(df):
    """
    Compute magnitude from X, Y, Z accelerometer data.
    """
    df = df.copy()
    df['magnitude'] = np.sqrt(df['X (m/s^2)']**2 +
                              df['Y (m/s^2)']**2 +
                              df['Z (m/s^2)']**2)
    return df