# csv_loader.py
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, List
import traceback

def compute_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Compute magnitude from acceleration columns with multiple fallbacks"""
    # Try standard names first
    if all(col in df.columns for col in ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]):
        df["magnitude"] = np.sqrt(
            df["X (m/s^2)"]**2 + 
            df["Y (m/s^2)"]**2 + 
            df["Z (m/s^2)"]**2
        )
        st.success("✅ Magnitude computed from standardized columns")
        return df
    
    # Try direct acceleration columns
    acceleration_cols = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'accelerationx' in col_lower or (col_lower.endswith('x') and 'acc' in col_lower):
            acceleration_cols['x'] = col
        elif 'accelerationy' in col_lower or (col_lower.endswith('y') and 'acc' in col_lower):
            acceleration_cols['y'] = col
        elif 'accelerationz' in col_lower or (col_lower.endswith('z') and 'acc' in col_lower):
            acceleration_cols['z'] = col
    
    if len(acceleration_cols) == 3:
        df["magnitude"] = np.sqrt(
            df[acceleration_cols['x']]**2 + 
            df[acceleration_cols['y']]**2 + 
            df[acceleration_cols['z']]**2
        )
        st.success(f"✅ Magnitude computed from: {acceleration_cols['x']}, {acceleration_cols['y']}, {acceleration_cols['z']}")
        return df
    
    st.error(f"❌ Could not find 3 acceleration columns for magnitude. Found: {list(acceleration_cols.keys())}")
    return df

def load_flexible_csv(uploaded_file, app_type="activity"):
    """
    Load CSV with flexible column naming.
    """
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        original_columns = df.columns.tolist()
        
        # ADD DEBUGGING
        st.write("🔍 **Debug: Raw CSV Info**")
        st.write(f"Original columns: {original_columns}")
        st.write(f"Number of rows: {len(df)}")
        
        # Show first few rows
        with st.expander("View raw data"):
            st.dataframe(df.head())
        
        # Define patterns - SIMPLIFIED for better matching
        PATTERNS = {
            'time': ['time', 'timestamp', 't', 'time(s)', 'time (s)', 'seconds'],
            
            # Accelerometer - SIMPLIFIED patterns
            'acc_x': ['accelerationx', 'acceleration_x', 'accx', 'ax', 'x_acc', 'acc_x', 'x'],
            'acc_y': ['accelerationy', 'acceleration_y', 'accy', 'ay', 'y_acc', 'acc_y', 'y'],
            'acc_z': ['accelerationz', 'acceleration_z', 'accz', 'az', 'z_acc', 'acc_z', 'z'],
        }
        
        if app_type != "activity":
            PATTERNS.update({
                'gyro_x': ['rotationratex', 'gyro_x', 'gyrox', 'rotation_x'],
                'gyro_y': ['rotationratey', 'gyro_y', 'gyroy', 'rotation_y'],
                'gyro_z': ['rotationratez', 'gyro_z', 'gyroz', 'rotation_z'],
                'pitch': ['pitch'],
                'roll': ['roll'],
                'yaw': ['yaw']
            })
        
        # Standard column names
        STANDARD_NAMES = {
            'time': "Time (s)",
            'acc_x': "X (m/s^2)",
            'acc_y': "Y (m/s^2)",
            'acc_z': "Z (m/s^2)",
        }
        
        if app_type != "activity":
            STANDARD_NAMES.update({
                'gyro_x': "Gyro_X",
                'gyro_y': "Gyro_Y",
                'gyro_z': "Gyro_Z",
                'pitch': "Pitch",
                'roll': "Roll",
                'yaw': "Yaw"
            })
        
        # NEW: Direct column detection (bypass pattern matching issues)
        found_columns = {}
        column_mapping = {}
        
        st.write("🔍 **Debug: Column Detection**")
        
        # Direct check for exact matches first
        for col in df.columns:
            col_lower = str(col).strip().lower()
            col_original = str(col).strip()
            
            st.write(f"Checking column: '{col_original}' (lower: '{col_lower}')")
            
            # Time columns
            if 'seconds_elapsed' in col_lower:
                found_columns['time'] = col_original
                column_mapping[col_original] = "Time (s)"
                st.write(f"  ✓ Matched as time column")
                continue
            
            # Acceleration columns
            if col_lower == 'accelerationx' or col_lower == 'acceleration_x':
                found_columns['acc_x'] = col_original
                column_mapping[col_original] = "X (m/s^2)"
                st.write(f"  ✓ Matched as acceleration X")
            elif col_lower == 'accelerationy' or col_lower == 'acceleration_y':
                found_columns['acc_y'] = col_original
                column_mapping[col_original] = "Y (m/s^2)"
                st.write(f"  ✓ Matched as acceleration Y")
            elif col_lower == 'accelerationz' or col_lower == 'acceleration_z':
                found_columns['acc_z'] = col_original
                column_mapping[col_original] = "Z (m/s^2)"
                st.write(f"  ✓ Matched as acceleration Z")
            
            # For multi-sensor
            if app_type != "activity":
                if 'rotationratex' in col_lower:
                    found_columns['gyro_x'] = col_original
                    column_mapping[col_original] = "Gyro_X"
                elif 'rotationratey' in col_lower:
                    found_columns['gyro_y'] = col_original
                    column_mapping[col_original] = "Gyro_Y"
                elif 'rotationratez' in col_lower:
                    found_columns['gyro_z'] = col_original
                    column_mapping[col_original] = "Gyro_Z"
                elif 'pitch' in col_lower:
                    found_columns['pitch'] = col_original
                    column_mapping[col_original] = "Pitch"
                elif 'roll' in col_lower:
                    found_columns['roll'] = col_original
                    column_mapping[col_original] = "Roll"
                elif 'yaw' in col_lower:
                    found_columns['yaw'] = col_original
                    column_mapping[col_original] = "Yaw"
        
        st.write(f"Found columns: {found_columns}")
        
        # Check minimum requirements
        if app_type == "activity":
            required = ['time', 'acc_x', 'acc_y', 'acc_z']
            missing = [col_type for col_type in required if col_type not in found_columns]
            
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.write("**Available columns:**")
                for col in df.columns:
                    st.write(f"- '{col}'")
                
                # Try alternative detection
                st.info("Trying alternative detection...")
                
                # Look for any acceleration-like columns
                acc_candidates = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'acc' in col_lower:
                        if 'x' in col_lower or col.endswith('X'):
                            acc_candidates['x'] = col
                        elif 'y' in col_lower or col.endswith('Y'):
                            acc_candidates['y'] = col
                        elif 'z' in col_lower or col.endswith('Z'):
                            acc_candidates['z'] = col
                
                st.write(f"Acceleration candidates: {acc_candidates}")
                
                if len(acc_candidates) == 3:
                    # Found all three!
                    for axis, col in acc_candidates.items():
                        column_mapping[col] = f"{axis.upper()} (m/s^2)"
                    # Find time column
                    for col in df.columns:
                        if any(time_word in str(col).lower() for time_word in ['time', 'second', 'elapsed']):
                            column_mapping[col] = "Time (s)"
                            break
                    
                    st.success("✅ Found acceleration columns through alternative detection!")
                else:
                    # Offer manual mapping
                    if st.checkbox("📋 Try manual column mapping?"):
                        return manual_column_mapping(df, app_type)
                    return None
        
        # Apply column renaming
        if column_mapping:
            df = df.rename(columns=column_mapping)
            st.success(f"✅ Renamed {len(column_mapping)} columns")
        else:
            st.warning("⚠️ No columns were renamed")
        
        # Show renamed columns
        st.write("🔍 **Debug: After renaming**")
        st.write(f"Columns: {df.columns.tolist()}")
        
        # Convert to numeric and clean
        numeric_cols_to_check = []
        if app_type == "activity":
            numeric_cols_to_check = ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
        else:
            numeric_cols_to_check = [col for col in df.columns if col in [
                "Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)",
                "Gyro_X", "Gyro_Y", "Gyro_Z", "Pitch", "Roll", "Yaw"
            ]]
        
        for col in numeric_cols_to_check:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in required columns
        if app_type == "activity":
            required_cols = [col for col in ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"] if col in df.columns]
        else:
            required_cols = [col for col in numeric_cols_to_check if col in df.columns]
        
        if required_cols:
            df = df.dropna(subset=required_cols)
        
        if df.empty:
            st.error("❌ No valid data after cleaning.")
            return None
        
        # Sort by time
        if "Time (s)" in df.columns:
            df = df.sort_values(by="Time (s)")
        
        st.info(f"📊 Loaded {len(df)} clean rows")
        
        # Compute magnitude
        df = compute_magnitude(df)
        
        # For multi-sensor data
        if app_type != "activity":
            if all(col in df.columns for col in ["Gyro_X", "Gyro_Y", "Gyro_Z"]):
                df["gyro_magnitude"] = np.sqrt(df["Gyro_X"]**2 + df["Gyro_Y"]**2 + df["Gyro_Z"]**2)
                st.success("✅ Computed gyroscope magnitude")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        st.code(traceback.format_exc())
        return None