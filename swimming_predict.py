import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import plotly.graph_objects as go
import plotly.express as px
import pickle
import tempfile
import os
from datetime import datetime
import matplotlib

# ==========================
# MAIN FUNCTION TO RUN SWIMMING APP
# ==========================
def run_swimming_app():
    """
    Main function to run the swimming prediction app.
    Called from app.py when user selects "Accelerometer + Gyroscope (Swimming)"
    """
    
    # Check for required modules
    try:
        import sklearn
    except ImportError:
        st.error("""
        ❌ **scikit-learn not found!**

        Please install required dependencies by running:
        ```
        pip install scikit-learn pandas numpy matplotlib plotly scipy
        ```
        """)
        return

    # ==========================
    # SWIMMING-SPECIFIC FUNCTIONS
    # ==========================
    def compute_magnitude_for_swimming(df):
        """Compute magnitude for acceleration and gyroscope for swimming data"""
        df = df.copy()
        
        # Standardize column names for swimming data
        column_mapping = {
            "seconds_elapsed": "Time (s)",
            "accelerationX": "Acc_X",
            "accelerationY": "Acc_Y", 
            "accelerationZ": "Acc_Z",
            "rotationRateX": "Gyro_X",
            "rotationRateY": "Gyro_Y",
            "rotationRateZ": "Gyro_Z",
            "pitch": "Pitch",
            "roll": "Roll",
            "yaw": "Yaw"
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Compute magnitudes
        if all(col in df.columns for col in ["Acc_X", "Acc_Y", "Acc_Z"]):
            df["acc_magnitude"] = np.sqrt(df["Acc_X"]**2 + df["Acc_Y"]**2 + df["Acc_Z"]**2)
        
        if all(col in df.columns for col in ["Gyro_X", "Gyro_Y", "Gyro_Z"]):
            df["gyro_magnitude"] = np.sqrt(df["Gyro_X"]**2 + df["Gyro_Y"]**2 + df["Gyro_Z"]**2)
        
        return df

    def extract_swimming_features(df, segment_duration):
        """Extract swimming-specific features (16 features)"""
        features = []
        
        # Get time column
        time_col = "Time (s)" if "Time (s)" in df.columns else "seconds_elapsed"
        
        if time_col not in df.columns:
            st.error(f"❌ Time column not found. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        start_time = df[time_col].min()
        end_time = df[time_col].max()
        
        for seg_start in np.arange(start_time, end_time, segment_duration):
            seg_end = seg_start + segment_duration
            
            seg_df = df[
                (df[time_col] >= seg_start) &
                (df[time_col] < seg_end)
            ]
            
            if seg_df.empty or len(seg_df) < 10:
                continue
            
            # Check if we have the required columns
            has_acc = "acc_magnitude" in seg_df.columns
            has_gyro = "gyro_magnitude" in seg_df.columns
            has_pitch = "Pitch" in seg_df.columns or "pitch" in seg_df.columns
            has_roll = "Roll" in seg_df.columns or "roll" in seg_df.columns
            has_gyro_z = "Gyro_Z" in seg_df.columns or "rotationRateZ" in seg_df.columns
            
            # Initialize feature dictionary
            feature_dict = {
                "segment_start": seg_start,
                "segment_end": seg_end,
                "duration": seg_end - seg_start,
                "sample_count": len(seg_df)
            }
            
            # ==========================
            # MOTION FEATURES (8 features)
            # ==========================
            if has_acc:
                acc_mag = seg_df["acc_magnitude"]
                feature_dict.update({
                    "acc_mean": acc_mag.mean(),
                    "acc_rms": np.sqrt(np.mean(acc_mag**2)),
                    "acc_std": acc_mag.std(),
                    "acc_p2p": acc_mag.max() - acc_mag.min(),
                })
            else:
                # If no acceleration data, fill with zeros
                feature_dict.update({
                    "acc_mean": 0,
                    "acc_rms": 0,
                    "acc_std": 0,
                    "acc_p2p": 0,
                })
            
            if has_gyro:
                gyro_mag = seg_df["gyro_magnitude"]
                feature_dict.update({
                    "gyro_mean": gyro_mag.mean(),
                    "gyro_rms": np.sqrt(np.mean(gyro_mag**2)),
                    "gyro_std": gyro_mag.std(),
                    "gyro_p2p": gyro_mag.max() - gyro_mag.min(),
                })
            else:
                # If no gyro data, fill with zeros
                feature_dict.update({
                    "gyro_mean": 0,
                    "gyro_rms": 0,
                    "gyro_std": 0,
                    "gyro_p2p": 0,
                })
            
            # ==========================
            # SHAPE FEATURES (8 features)
            # ==========================
            if has_pitch:
                pitch_col = "Pitch" if "Pitch" in seg_df.columns else "pitch"
                pitch_seg = seg_df[pitch_col].values
                feature_dict.update({
                    "pitch_kurtosis": kurtosis(pitch_seg) if len(pitch_seg) > 3 else 0,
                    "pitch_skewness": skew(pitch_seg) if len(pitch_seg) > 3 else 0,
                    "pitch_peak_count": len(find_peaks(pitch_seg)[0]) if len(pitch_seg) > 1 else 0,
                })
            else:
                feature_dict.update({
                    "pitch_kurtosis": 0,
                    "pitch_skewness": 0,
                    "pitch_peak_count": 0,
                })
            
            if has_roll:
                roll_col = "Roll" if "Roll" in seg_df.columns else "roll"
                roll_seg = seg_df[roll_col].values
                # Roll asymmetry
                if len(roll_seg) > 0:
                    # Separate positive and negative rolls
                    pos_rolls = roll_seg[roll_seg > 0]
                    neg_rolls = roll_seg[roll_seg < 0]
                    
                    if len(pos_rolls) > 0 and len(neg_rolls) > 0:
                        roll_asym = np.mean(pos_rolls) - np.mean(np.abs(neg_rolls))
                    elif len(pos_rolls) > 0:
                        roll_asym = np.mean(pos_rolls)
                    elif len(neg_rolls) > 0:
                        roll_asym = -np.mean(np.abs(neg_rolls))
                    else:
                        roll_asym = 0
                else:
                    roll_asym = 0
                feature_dict["roll_asymmetry"] = roll_asym
            else:
                feature_dict["roll_asymmetry"] = 0
            
            # Stroke features from gyro Z
            if has_gyro_z:
                gyro_z_col = "Gyro_Z" if "Gyro_Z" in seg_df.columns else "rotationRateZ"
                gyro_z_seg = seg_df[gyro_z_col].values
                
                # Stroke frequency
                if len(gyro_z_seg) > 0:
                    peaks, _ = find_peaks(gyro_z_seg)
                    stroke_freq = len(peaks) / segment_duration
                else:
                    stroke_freq = 0
                
                # Stroke rhythm regularity
                if len(peaks) > 2:
                    intervals = np.diff(peaks)
                    if np.mean(intervals) > 0:
                        rhythm_cv = np.std(intervals) / np.mean(intervals)
                    else:
                        rhythm_cv = 0
                else:
                    rhythm_cv = 0
                
                feature_dict.update({
                    "stroke_frequency": stroke_freq,
                    "stroke_rhythm_cv": rhythm_cv,
                })
            else:
                feature_dict.update({
                    "stroke_frequency": 0,
                    "stroke_rhythm_cv": 0,
                })
            
            # Additional gyro shape features
            if has_gyro:
                gyro_mag = seg_df["gyro_magnitude"].values
                if len(gyro_mag) > 3:
                    feature_dict.update({
                        "gyro_kurtosis": kurtosis(gyro_mag),
                        "gyro_skewness": skew(gyro_mag),
                    })
                else:
                    feature_dict.update({
                        "gyro_kurtosis": 0,
                        "gyro_skewness": 0,
                    })
            else:
                feature_dict.update({
                    "gyro_kurtosis": 0,
                    "gyro_skewness": 0,
                })
            
            features.append(feature_dict)
        
        if features:
            features_df = pd.DataFrame(features)
            return features_df
        else:
            return pd.DataFrame()

    # ==========================
    # PAGE CONFIG
    # ==========================
    # st.set_page_config(
    #     page_title="Swimming Stroke Prediction",
    #     layout="wide",
    #     page_icon="assets/favicon.ico"
    # )

    st.title("🏊‍♂️ Swimming Stroke Recognition: Model Comparison Predictor")
    st.markdown("""
    **Academic Tool:** Load trained swimming models (KNN and Random Forest), compare predictions, and analyze algorithm performance.
    Upload swimming models to see where they agree/disagree on new swimming data.
    """)

    # ==========================
    # SESSION STATE
    # ==========================
    if 'swimming_model1' not in st.session_state:
        st.session_state.swimming_model1 = None
        st.session_state.swimming_model1_type = None

    if 'swimming_model2' not in st.session_state:
        st.session_state.swimming_model2 = None
        st.session_state.swimming_model2_type = None

    if 'swimming_new_data' not in st.session_state:
        st.session_state.swimming_new_data = None

    if 'swimming_predictions_df' not in st.session_state:
        st.session_state.swimming_predictions_df = None

    if 'swimming_features_df' not in st.session_state:
        st.session_state.swimming_features_df = None

    # ==========================
    # SIDEBAR - MODEL UPLOAD
    # ==========================
    with st.sidebar:
        st.image("assets/logo.png", width=150)
        st.markdown("---")

        st.header("🔧 Model Configuration")

        # MODEL 1 UPLOAD
        st.subheader("1. Upload Swimming Model 1")
        model1_file = st.file_uploader(
            "📤 Upload first swimming model (.pkl)",
            type=["pkl"],
            key="swimming_model1_uploader",
            help="Upload KNN or Random Forest swimming model (16 features)"
        )

        if model1_file is not None and st.session_state.swimming_model1 is None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(model1_file.getvalue())
                    tmp_path = tmp_file.name

                with open(tmp_path, 'rb') as f:
                    loaded_data = pickle.load(f)

                os.unlink(tmp_path)

                st.session_state.swimming_model1 = loaded_data
                st.session_state.swimming_model1_type = "KNN" if 'scaler' in loaded_data else "Random Forest"

                st.success(f"✅ Model 1 Loaded ({st.session_state.swimming_model1_type})")

                with st.expander("📋 Model 1 Details", expanded=True):
                    st.write(f"**Type:** {st.session_state.swimming_model1_type}")
                    st.write(f"**Accuracy:** {loaded_data['accuracy']:.3f}")
                    st.write(f"**Features:** {len(loaded_data['feature_names'])} features")
                    st.write(f"**Classes:** {', '.join(loaded_data['classes'])}")
                    if 'training_date' in loaded_data:
                        st.write(f"**Trained:** {loaded_data['training_date']}")
                    if st.session_state.swimming_model1_type == "KNN":
                        st.write(f"**Optimal k:** {loaded_data.get('optimal_k', 'N/A')}")
                    else:
                        st.write(f"**Trees:** {loaded_data.get('n_estimators', 'N/A')}")
                    
                    # Show feature names for debugging
                    if 'feature_names' in loaded_data:
                        st.write(f"**First 5 Features:** {', '.join(loaded_data['feature_names'][:5])}")

            except Exception as e:
                st.error(f"❌ Error loading Model 1: {str(e)}")

        # MODEL 2 UPLOAD
        st.subheader("2. Upload Swimming Model 2")
        model2_file = st.file_uploader(
            "📤 Upload second swimming model (.pkl)",
            type=["pkl"],
            key="swimming_model2_uploader",
            help="Upload a different model for comparison"
        )

        if model2_file is not None and st.session_state.swimming_model2 is None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(model2_file.getvalue())
                    tmp_path = tmp_file.name

                with open(tmp_path, 'rb') as f:
                    loaded_data = pickle.load(f)

                os.unlink(tmp_path)

                st.session_state.swimming_model2 = loaded_data
                st.session_state.swimming_model2_type = "KNN" if 'scaler' in loaded_data else "Random Forest"

                st.success(f"✅ Model 2 Loaded ({st.session_state.swimming_model2_type})")

                with st.expander("📋 Model 2 Details", expanded=True):
                    st.write(f"**Type:** {st.session_state.swimming_model2_type}")
                    st.write(f"**Accuracy:** {loaded_data['accuracy']:.3f}")
                    st.write(f"**Features:** {len(loaded_data['feature_names'])} features")
                    st.write(f"**Classes:** {', '.join(loaded_data['classes'])}")
                    if 'training_date' in loaded_data:
                        st.write(f"**Trained:** {loaded_data['training_date']}")
                    if st.session_state.swimming_model2_type == "KNN":
                        st.write(f"**Optimal k:** {loaded_data.get('optimal_k', 'N/A')}")
                    else:
                        st.write(f"**Trees:** {loaded_data.get('n_estimators', 'N/A')}")
                    
                    # Show feature names for debugging
                    if 'feature_names' in loaded_data:
                        st.write(f"**First 5 Features:** {', '.join(loaded_data['feature_names'][:5])}")

            except Exception as e:
                st.error(f"❌ Error loading Model 2: {str(e)}")

        st.markdown("---")

        # DATA PROCESSING SETTINGS
        st.subheader("3. Processing Settings")

        segment_duration = st.slider(
            "Segment Duration (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Window size for feature extraction",
            key="swimming_segment_duration"
        )

        confidence_threshold = st.slider(
            "Low Confidence Threshold (%)",
            min_value=50,
            max_value=90,
            value=70,
            help="Flag predictions below this confidence",
            key="swimming_confidence_threshold"
        ) / 100

        # Reset button
        if st.button("🔄 Reset All", type="secondary", key="swimming_reset"):
            for key in ['swimming_model1', 'swimming_model1_type', 'swimming_model2', 'swimming_model2_type',
                       'swimming_new_data', 'swimming_predictions_df', 'swimming_features_df']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Model comparison info
        if st.session_state.swimming_model1 is not None and st.session_state.swimming_model2 is not None:
            st.markdown("---")
            st.subheader("📊 Model Comparison Ready")

            model1_acc = st.session_state.swimming_model1['accuracy']
            model2_acc = st.session_state.swimming_model2['accuracy']

            if model1_acc > model2_acc:
                st.info(f"**Model 1 has higher accuracy** (+{(model1_acc - model2_acc) * 100:.1f}%)")
            elif model2_acc > model1_acc:
                st.info(f"**Model 2 has higher accuracy** (+{(model2_acc - model1_acc) * 100:.1f}%)")
            else:
                st.info("**Models have equal accuracy**")

            if st.session_state.swimming_model1_type == st.session_state.swimming_model2_type:
                st.warning("⚠️ Both models are same type. Consider uploading different algorithms.")

    # ==========================
    # MAIN INTERFACE
    # ==========================
    # Check if models are loaded
    if st.session_state.swimming_model1 is None or st.session_state.swimming_model2 is None:
        # Show instructions if models not loaded
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 How to Use:

            1. **Upload Two Swimming Models** (sidebar):
               - Model 1: KNN or Random Forest (16 features)
               - Model 2: Different algorithm recommended

            2. **Upload New Swimming CSV Data**:
               - Apple Watch format: seconds_elapsed, accelerationX/Y/Z, rotationRateX/Y/Z, pitch, roll

            3. **Compare Predictions**:
               - See where models agree/disagree on swimming strokes
               - Analyze algorithm differences
               - Export comparison results
            """)

        with col2:
            st.markdown("""
            ### 🎯 What You'll See:

            **📊 Agreement Analysis:**
            - Percentage of matching predictions
            - Confidence comparison

            **🔍 Disagreement Analysis:**
            - Segments where models disagree
            - Feature analysis of disagreements
            - Confidence scores for each

            **📈 Visual Comparisons:**
            - Side-by-side timelines
            - Confidence distributions
            - Feature space visualization
            """)

        st.markdown("---")

        # Example models info
        with st.expander("💡 Swimming Model Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🏊‍♂️ Swimming KNN Model:**
                - 8 motion features only
                - Acceleration and gyroscope magnitudes
                - Fast predictions
                - Sensitive to feature scaling
                """)
            with col2:
                st.markdown("""
                **🌲 Swimming Random Forest Model:**
                - 16 features (8 motion + 8 shape)
                - Includes pitch, roll, stroke frequency
                - Handles complex swimming patterns
                - Provides feature importance
                """)

        st.info("👈 **Start by uploading two swimming models in the sidebar!**")
        return

    # Models are loaded - show main interface
    st.success(f"✅ Ready: Model 1 ({st.session_state.swimming_model1_type}) vs Model 2 ({st.session_state.swimming_model2_type})")

    # ==========================
    # DATA UPLOAD SECTION
    # ==========================
    st.header("📤 Step 3: Upload New Swimming Data")

    new_csv = st.file_uploader(
        "Upload swimming CSV file for prediction",
        type=["csv"],
        help="Apple Watch format: seconds_elapsed, accelerationX/Y/Z, rotationRateX/Y/Z, pitch, roll, yaw",
        key="swimming_csv_uploader"
    )

    if new_csv is None:
        st.info("📝 Upload a swimming CSV file to start prediction comparison")
        return

    try:
        # Load and process new data
        new_df = pd.read_csv(new_csv)
        
        # Check for required swimming columns
        required_cols_options = [
            ["seconds_elapsed", "accelerationX", "accelerationY", "accelerationZ"],
            ["Time (s)", "Acc_X", "Acc_Y", "Acc_Z"],
            ["time", "ax", "ay", "az"],
            ["timestamp", "accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"]
        ]
        
        has_required_cols = False
        for cols in required_cols_options:
            if all(col in new_df.columns for col in cols):
                has_required_cols = True
                break
        
        if not has_required_cols:
            st.error("""
            ❌ CSV must contain swimming sensor data. Expected columns like:
            - seconds_elapsed, accelerationX, accelerationY, accelerationZ
            - rotationRateX, rotationRateY, rotationRateZ (optional)
            - pitch, roll (optional)
            
            **Available columns:** """ + ', '.join(new_df.columns))
            return
        
        # Compute magnitudes for swimming data
        new_df = compute_magnitude_for_swimming(new_df)
        st.session_state.swimming_new_data = new_df
        
        # Show data preview
        with st.expander("📄 Data Preview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", len(new_df))
                time_col = "Time (s)" if "Time (s)" in new_df.columns else "seconds_elapsed"
                if time_col in new_df.columns:
                    duration = new_df[time_col].max() - new_df[time_col].min()
                    st.metric("Duration", f"{duration:.1f}s")
                else:
                    st.metric("Duration", "N/A")
            
            with col2:
                if time_col in new_df.columns:
                    st.metric("Time Range", f"{new_df[time_col].min():.1f}s to {new_df[time_col].max():.1f}s")
                else:
                    st.metric("Time Range", "N/A")
            
            with col3:
                if time_col in new_df.columns and len(new_df) > 1:
                    sampling_rate = 1 / np.mean(np.diff(new_df[time_col]))
                    st.metric("Sampling Rate", f"{sampling_rate:.1f} Hz")
                else:
                    st.metric("Sampling Rate", "N/A")
            
            # Show available sensors
            sensor_info = []
            if "acc_magnitude" in new_df.columns:
                sensor_info.append("📱 Accelerometer")
            if "gyro_magnitude" in new_df.columns:
                sensor_info.append("🔄 Gyroscope")
            if "Pitch" in new_df.columns or "pitch" in new_df.columns:
                sensor_info.append("📐 Pitch")
            if "Roll" in new_df.columns or "roll" in new_df.columns:
                sensor_info.append("🔄 Roll")
            
            if sensor_info:
                st.write(f"**Available Sensors:** {', '.join(sensor_info)}")
            
            st.dataframe(new_df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading swimming data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return

    # ==========================
    # PROCESSING AND PREDICTION
    # ==========================
    if st.button("⚡ Compare Swimming Model Predictions", type="primary", use_container_width=True, key="swimming_predict_button"):
        with st.spinner("Extracting swimming features and comparing predictions..."):
            try:
                # Extract swimming-specific features (16 features)
                features_df = extract_swimming_features(st.session_state.swimming_new_data, segment_duration)
                
                if features_df.empty:
                    st.warning("⚠️ No complete segments found in the data.")
                    return
                
                st.session_state.swimming_features_df = features_df
                
                # ==========================
                # MAKE PREDICTIONS WITH BOTH MODELS
                # ==========================
                predictions_data = []
                
                for idx, row in features_df.iterrows():
                    seg_data = {
                        "segment_id": idx + 1,
                        "start_time": row["segment_start"],
                        "end_time": row["segment_end"],
                        "duration": row["duration"],
                        "sample_count": row["sample_count"]
                    }
                    
                    # MODEL 1 PREDICTION
                    model1_data = st.session_state.swimming_model1
                    feature_names1 = model1_data['feature_names']
                    
                    # Select features for model 1 - handle missing features
                    X1_features = []
                    for feat in feature_names1:
                        if feat in row.index:
                            X1_features.append(row[feat])
                        else:
                            # If feature is missing, use 0 (or appropriate default)
                            X1_features.append(0.0)
                    
                    X1 = np.array(X1_features).reshape(1, -1)
                    
                    # Scale if KNN
                    if st.session_state.swimming_model1_type == "KNN":
                        X1_scaled = model1_data['scaler'].transform(X1)
                        pred1 = model1_data['model'].predict(X1_scaled)[0]
                        
                        # Get confidence (distance to nearest neighbors)
                        distances, indices = model1_data['model'].kneighbors(X1_scaled)
                        confidence1 = 1 / (1 + distances[0][0])  # Inverse distance to closest neighbor
                    else:
                        pred1 = model1_data['model'].predict(X1)[0]
                        # RF confidence (probability)
                        probs = model1_data['model'].predict_proba(X1)[0]
                        class_idx = list(model1_data['classes']).index(pred1)
                        confidence1 = probs[class_idx]
                    
                    seg_data.update({
                        "model1_pred": pred1,
                        "model1_conf": float(confidence1),
                        "model1_low_conf": confidence1 < confidence_threshold
                    })
                    
                    # MODEL 2 PREDICTION
                    model2_data = st.session_state.swimming_model2
                    feature_names2 = model2_data['feature_names']
                    
                    # Select features for model 2 - handle missing features
                    X2_features = []
                    for feat in feature_names2:
                        if feat in row.index:
                            X2_features.append(row[feat])
                        else:
                            X2_features.append(0.0)
                    
                    X2 = np.array(X2_features).reshape(1, -1)
                    
                    # Scale if KNN
                    if st.session_state.swimming_model2_type == "KNN":
                        X2_scaled = model2_data['scaler'].transform(X2)
                        pred2 = model2_data['model'].predict(X2_scaled)[0]
                        
                        # Get confidence
                        distances, indices = model2_data['model'].kneighbors(X2_scaled)
                        confidence2 = 1 / (1 + distances[0][0])
                    else:
                        pred2 = model2_data['model'].predict(X2)[0]
                        probs = model2_data['model'].predict_proba(X2)[0]
                        class_idx = list(model2_data['classes']).index(pred2)
                        confidence2 = probs[class_idx]
                    
                    seg_data.update({
                        "model2_pred": pred2,
                        "model2_conf": float(confidence2),
                        "model2_low_conf": confidence2 < confidence_threshold
                    })
                    
                    # COMPARISON METRICS
                    seg_data.update({
                        "agree": pred1 == pred2,
                        "conf_diff": confidence1 - confidence2,
                        "abs_conf_diff": abs(confidence1 - confidence2),
                        "avg_conf": (confidence1 + confidence2) / 2
                    })
                    
                    predictions_data.append(seg_data)
                
                predictions_df = pd.DataFrame(predictions_data)
                st.session_state.swimming_predictions_df = predictions_df
                
                # ==========================
                # DYNAMIC COLOR MAPPING FOR SWIMMING STROKES
                # ==========================
                # Get all unique strokes from both models
                all_strokes = set()
                if 'classes' in st.session_state.swimming_model1:
                    all_strokes.update(st.session_state.swimming_model1['classes'])
                if 'classes' in st.session_state.swimming_model2:
                    all_strokes.update(st.session_state.swimming_model2['classes'])
                
                # Also get strokes from predictions
                all_strokes.update(predictions_df['model1_pred'].unique())
                all_strokes.update(predictions_df['model2_pred'].unique())
                
                all_strokes = list(all_strokes)
                
                # Swimming-specific colors
                swimming_colors_matplotlib = {
                    "butterfly": "tab:blue",
                    "backstroke": "tab:green", 
                    "breaststroke": "tab:orange",
                    "freestyle": "tab:red"
                }
                
                swimming_colors_plotly = {
                    "butterfly": "blue",
                    "backstroke": "green",
                    "breaststroke": "orange", 
                    "freestyle": "red"
                }
                
                # Create color mappings for both matplotlib and plotly
                matplotlib_stroke_colors = {}
                plotly_stroke_colors = {}
                
                for i, stroke in enumerate(all_strokes):
                    # Use swimming colors if available, otherwise default
                    matplotlib_stroke_colors[stroke] = swimming_colors_matplotlib.get(
                        stroke, 
                        f"tab:{['blue', 'green', 'orange', 'red', 'purple', 'brown'][i % 6]}"
                    )
                    plotly_stroke_colors[stroke] = swimming_colors_plotly.get(
                        stroke,
                        ['blue', 'green', 'orange', 'red', 'purple', 'brown'][i % 6]
                    )
                
                # ==========================
                # DISPLAY COMPARISON RESULTS
                # ==========================
                st.header("📊 Model Comparison Results")
                
                # SUMMARY STATISTICS
                total_segments = len(predictions_df)
                agreement_count = predictions_df["agree"].sum()
                agreement_rate = agreement_count / total_segments
                
                model1_low_conf = predictions_df["model1_low_conf"].sum()
                model2_low_conf = predictions_df["model2_low_conf"].sum()
                
                avg_conf1 = predictions_df["model1_conf"].mean()
                avg_conf2 = predictions_df["model2_conf"].mean()
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Segments", total_segments)
                
                with col2:
                    st.metric("Agreement Rate", f"{agreement_rate:.1%}")
                
                with col3:
                    conf_diff = avg_conf1 - avg_conf2
                    st.metric("Avg Confidence Diff",
                              f"{abs(conf_diff):.3f}",
                              delta=f"{'Model 1' if conf_diff > 0 else 'Model 2'} higher",
                              delta_color="normal")
                
                with col4:
                    st.metric("Disagreements", total_segments - agreement_count)
                
                # ==========================
                # VISUALIZATION 1: AGREEMENT ANALYSIS
                # ==========================
                st.subheader("📈 Agreement Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Confidence", "Disagreements", "Feature Analysis"])
                
                with tab1:
                    # Timeline comparison
                    fig_timeline = go.Figure()
                    
                    # Add Model 1 predictions
                    for stroke in predictions_df['model1_pred'].unique():
                        stroke_data = predictions_df[predictions_df['model1_pred'] == stroke]
                        if not stroke_data.empty:
                            fig_timeline.add_trace(go.Scatter(
                                x=stroke_data['start_time'],
                                y=['Model 1'] * len(stroke_data),
                                mode='markers',
                                name=f'Model 1: {stroke}',
                                marker=dict(
                                    size=10,
                                    color=plotly_stroke_colors.get(stroke, 'gray'),
                                    symbol='circle',
                                    opacity=0.7
                                ),
                                hovertemplate='<b>Model 1</b><br>' +
                                              'Stroke: %{text}<br>' +
                                              'Time: %{x:.1f}s<br>' +
                                              'Confidence: %{customdata:.1%}<extra></extra>',
                                text=[stroke] * len(stroke_data),
                                customdata=stroke_data['model1_conf']
                            ))
                    
                    # Add Model 2 predictions
                    for stroke in predictions_df['model2_pred'].unique():
                        stroke_data = predictions_df[predictions_df['model2_pred'] == stroke]
                        if not stroke_data.empty:
                            fig_timeline.add_trace(go.Scatter(
                                x=stroke_data['start_time'],
                                y=['Model 2'] * len(stroke_data),
                                mode='markers',
                                name=f'Model 2: {stroke}',
                                marker=dict(
                                    size=10,
                                    color=plotly_stroke_colors.get(stroke, 'gray'),
                                    symbol='square',
                                    opacity=0.7
                                ),
                                hovertemplate='<b>Model 2</b><br>' +
                                              'Stroke: %{text}<br>' +
                                              'Time: %{x:.1f}s<br>' +
                                              'Confidence: %{customdata:.1%}<extra></extra>',
                                text=[stroke] * len(stroke_data),
                                customdata=stroke_data['model2_conf']
                            ))
                    
                    # Highlight disagreements
                    disagreements = predictions_df[~predictions_df['agree']]
                    if not disagreements.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=disagreements['start_time'],
                            y=['Disagreement'] * len(disagreements),
                            mode='markers',
                            name='Disagreement',
                            marker=dict(
                                size=12,
                                color='red',
                                symbol='x',
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='<b>DISAGREEMENT</b><br>' +
                                          'Time: %{x:.1f}s<br>' +
                                          'Model 1: %{customdata[0]} (%{customdata[1]:.1%})<br>' +
                                          'Model 2: %{customdata[2]} (%{customdata[3]:.1%})<extra></extra>',
                            customdata=np.column_stack([
                                disagreements['model1_pred'],
                                disagreements['model1_conf'],
                                disagreements['model2_pred'],
                                disagreements['model2_conf']
                            ])
                        ))
                    
                    fig_timeline.update_layout(
                        title='Swimming Stroke Predictions Timeline',
                        xaxis_title='Time (seconds)',
                        yaxis=dict(
                            tickmode='array',
                            tickvals=['Model 1', 'Model 2', 'Disagreement'],
                            ticktext=['Model 1', 'Model 2', 'Disagreements']
                        ),
                        height=400,
                        hovermode='closest',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with tab2:
                    # Confidence comparison
                    fig_conf = go.Figure()
                    
                    # Model 1 confidence distribution
                    fig_conf.add_trace(go.Histogram(
                        x=predictions_df['model1_conf'],
                        name='Model 1',
                        nbinsx=20,
                        opacity=0.7,
                        marker_color='blue'
                    ))
                    
                    # Model 2 confidence distribution
                    fig_conf.add_trace(go.Histogram(
                        x=predictions_df['model2_conf'],
                        name='Model 2',
                        nbinsx=20,
                        opacity=0.7,
                        marker_color='green'
                    ))
                    
                    # Add vertical line for threshold
                    fig_conf.add_vline(
                        x=confidence_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {confidence_threshold:.0%}",
                        annotation_position="top right"
                    )
                    
                    fig_conf.update_layout(
                        title='Confidence Distribution Comparison',
                        xaxis_title='Confidence',
                        yaxis_title='Count',
                        barmode='overlay',
                        height=400
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Confidence statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model 1 Avg Confidence", f"{avg_conf1:.3f}")
                        st.metric("Model 1 Low Confidence", f"{model1_low_conf} segments")
                    with col2:
                        st.metric("Model 2 Avg Confidence", f"{avg_conf2:.3f}")
                        st.metric("Model 2 Low Confidence", f"{model2_low_conf} segments")
                
                with tab3:
                    # Detailed disagreement analysis
                    disagreements_df = predictions_df[~predictions_df['agree']].copy()
                    
                    if not disagreements_df.empty:
                        st.write(f"**Found {len(disagreements_df)} disagreements:**")
                        
                        # Create detailed table
                        display_cols = [
                            'segment_id', 'start_time', 'end_time',
                            'model1_pred', 'model1_conf',
                            'model2_pred', 'model2_conf',
                            'conf_diff', 'avg_conf'
                        ]
                        
                        # Format for display
                        display_df = disagreements_df[display_cols].copy()
                        display_df['model1_conf'] = display_df['model1_conf'].apply(lambda x: f"{x:.1%}")
                        display_df['model2_conf'] = display_df['model2_conf'].apply(lambda x: f"{x:.1%}")
                        display_df['conf_diff'] = display_df['conf_diff'].apply(lambda x: f"{x:+.3f}")
                        display_df['avg_conf'] = display_df['avg_conf'].apply(lambda x: f"{x:.1%}")
                        
                        display_df.columns = [
                            'Segment', 'Start (s)', 'End (s)',
                            'Model 1 Pred', 'Model 1 Conf',
                            'Model 2 Pred', 'Model 2 Conf',
                            'Conf Diff', 'Avg Conf'
                        ]
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Disagreement patterns
                        st.subheader("Disagreement Patterns")
                        
                        # Create pattern matrix
                        pattern_counts = pd.crosstab(
                            disagreements_df['model1_pred'],
                            disagreements_df['model2_pred']
                        )
                        
                        fig_pattern = go.Figure(data=go.Heatmap(
                            z=pattern_counts.values,
                            x=pattern_counts.columns,
                            y=pattern_counts.index,
                            colorscale='Reds',
                            text=pattern_counts.values,
                            texttemplate='%{text}',
                            textfont={"size": 12},
                            hoverongaps=False
                        ))
                        
                        fig_pattern.update_layout(
                            title='Disagreement Patterns (Model 1 → Model 2)',
                            xaxis_title='Model 2 Prediction',
                            yaxis_title='Model 1 Prediction',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pattern, use_container_width=True)
                    else:
                        st.success("✅ Perfect agreement! No disagreements found.")
                
                with tab4:
                    # Feature analysis of disagreements
                    if not disagreements_df.empty:
                        # Merge features with predictions
                        analysis_df = pd.merge(
                            disagreements_df,
                            features_df,
                            left_on='segment_id',
                            right_index=True,
                            how='left'
                        )
                        
                        # Select key swimming features for analysis
                        key_features = ['acc_mean', 'acc_rms', 'gyro_mean', 'gyro_rms',
                                       'stroke_frequency', 'roll_asymmetry', 'pitch_peak_count']
                        
                        # Only use features that exist
                        key_features = [f for f in key_features if f in analysis_df.columns]
                        
                        st.write("**Feature Statistics for Disagreement Segments:**")
                        
                        # Calculate statistics
                        stats_data = []
                        for feature in key_features:
                            stats_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Mean': analysis_df[feature].mean(),
                                'Std': analysis_df[feature].std(),
                                'Min': analysis_df[feature].min(),
                                'Max': analysis_df[feature].max()
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Feature comparison plot
                        fig_features = go.Figure()
                        
                        for idx, row in analysis_df.iterrows():
                            fig_features.add_trace(go.Scatter(
                                x=key_features,
                                y=[row[feat] for feat in key_features],
                                mode='lines+markers',
                                name=f"Seg {row['segment_id']} ({row['start_time']:.1f}s)",
                                hovertemplate='<b>Segment %{customdata[0]}</b><br>' +
                                              '%{x}: %{y:.3f}<br>' +
                                              'Model 1: %{customdata[1]}<br>' +
                                              'Model 2: %{customdata[2]}<extra></extra>',
                                customdata=[[row['segment_id']] * len(key_features),
                                            [row['model1_pred']] * len(key_features),
                                            [row['model2_pred']] * len(key_features)]
                            ))
                        
                        fig_features.update_layout(
                            title='Swimming Feature Values for Disagreement Segments',
                            xaxis_title='Feature',
                            yaxis_title='Value',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_features, use_container_width=True)
                    else:
                        st.info("No disagreements to analyze.")
                
                # ==========================
                # VISUALIZATION 2: RAW SIGNAL WITH PREDICTIONS
                # ==========================
                st.subheader("📈 Swimming Signal with Predictions")
                
                # Create figure
                fig_signal, ax = plt.subplots(figsize=(12, 6))
                
                # Plot raw acceleration signal
                time_col = "Time (s)" if "Time (s)" in st.session_state.swimming_new_data.columns else "seconds_elapsed"
                
                if "acc_magnitude" in st.session_state.swimming_new_data.columns:
                    ax.plot(st.session_state.swimming_new_data[time_col], st.session_state.swimming_new_data["acc_magnitude"],
                            color="blue", alpha=0.6, linewidth=0.8, label="Acceleration Magnitude")
                
                if "gyro_magnitude" in st.session_state.swimming_new_data.columns:
                    ax.plot(st.session_state.swimming_new_data[time_col], st.session_state.swimming_new_data["gyro_magnitude"],
                            color="green", alpha=0.6, linewidth=0.8, label="Gyroscope Magnitude")
                
                # Add colored regions for Model 1 predictions using dynamic colors
                for _, row in predictions_df.iterrows():
                    stroke = row['model1_pred']
                    color = matplotlib_stroke_colors.get(stroke, 'gray')
                    
                    # Different opacity for disagreements
                    alpha = 0.15 if row['agree'] else 0.3
                    
                    ax.axvspan(row['start_time'], row['end_time'],
                               alpha=alpha, color=color)
                
                # Custom legend
                from matplotlib.patches import Patch
                
                legend_elements = []
                # Only show strokes that actually appear in predictions
                predicted_strokes = predictions_df['model1_pred'].unique()
                for stroke in predicted_strokes:
                    color = matplotlib_stroke_colors.get(stroke, 'gray')
                    display_name = stroke.replace('_', ' ').title()
                    legend_elements.append(
                        Patch(facecolor=color, alpha=0.2,
                              label=f"{display_name} (Model 1)")
                    )
                
                # Add disagreement markers
                disagreements = predictions_df[~predictions_df['agree']]
                if not disagreements.empty:
                    for _, row in disagreements.iterrows():
                        mid_time = (row['start_time'] + row['end_time']) / 2
                        # Find closest signal value
                        time_idx = np.abs(st.session_state.swimming_new_data[time_col] - mid_time).argmin()
                        if "acc_magnitude" in st.session_state.swimming_new_data.columns:
                            signal_val = st.session_state.swimming_new_data.loc[time_idx, "acc_magnitude"]
                        else:
                            signal_val = 0
                        ax.plot(mid_time, signal_val, 'rx', markersize=10,
                                label='Disagreement' if _ == disagreements.index[0] else "")
                
                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper left',
                              bbox_to_anchor=(1.01, 1))
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Magnitude")
                ax.grid(True, alpha=0.3)
                ax.set_title("Swimming Signals with Model 1 Predictions (Red X = Disagreement)")
                
                st.pyplot(fig_signal)
                
                # ==========================
                # PERFORMANCE METRICS BY STROKE
                # ==========================
                st.subheader("🎯 Performance by Swimming Stroke")
                
                # Calculate agreement by stroke
                stroke_agreement = []
                for stroke in predictions_df['model1_pred'].unique():
                    stroke_data = predictions_df[predictions_df['model1_pred'] == stroke]
                    if len(stroke_data) > 0:
                        agreement_rate_stroke = stroke_data['agree'].mean()
                        avg_conf1_stroke = stroke_data['model1_conf'].mean()
                        avg_conf2_stroke = stroke_data['model2_conf'].mean()
                        
                        stroke_agreement.append({
                            'Stroke': stroke.replace('_', ' ').title(),
                            'Segments': len(stroke_data),
                            'Agreement Rate': agreement_rate_stroke,
                            'Model 1 Avg Conf': avg_conf1_stroke,
                            'Model 2 Avg Conf': avg_conf2_stroke,
                            'Conf Difference': avg_conf1_stroke - avg_conf2_stroke
                        })
                
                agreement_df = pd.DataFrame(stroke_agreement)
                
                if not agreement_df.empty:
                    # Display table
                    st.dataframe(agreement_df, use_container_width=True)
                    
                    # Agreement rate by stroke bar chart
                    fig_agreement = go.Figure(data=[
                        go.Bar(
                            x=agreement_df['Stroke'],
                            y=agreement_df['Agreement Rate'],
                            text=agreement_df['Agreement Rate'].apply(lambda x: f"{x:.1%}"),
                            textposition='auto',
                            marker_color='lightblue'
                        )
                    ])
                    
                    fig_agreement.update_layout(
                        title='Agreement Rate by Swimming Stroke (Model 1 Perspective)',
                        yaxis_title='Agreement Rate',
                        yaxis=dict(tickformat='.0%'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_agreement, use_container_width=True)
                
                # ==========================
                # DOWNLOAD SECTION
                # ==========================
                st.markdown("---")
                st.subheader("💾 Download Comparison Results")
                
                # Create download dataframe
                download_data = predictions_df.copy()
                
                # Add swimming feature columns
                swimming_feature_cols = [
                    'acc_mean', 'acc_rms', 'acc_std', 'acc_p2p',
                    'gyro_mean', 'gyro_rms', 'gyro_std', 'gyro_p2p',
                    'pitch_kurtosis', 'pitch_skewness', 'pitch_peak_count',
                    'roll_asymmetry', 'stroke_frequency', 'stroke_rhythm_cv',
                    'gyro_kurtosis', 'gyro_skewness'
                ]
                
                for col in swimming_feature_cols:
                    if col in features_df.columns:
                        # Map features by segment_id
                        download_data[col] = download_data['segment_id'].apply(
                            lambda x: features_df.loc[x - 1, col] if x - 1 in features_df.index else None
                        )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Detailed predictions
                    csv_data = download_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Detailed Results",
                        data=csv_data,
                        file_name=f"swimming_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Complete dataset with all predictions and swimming features"
                    )
                
                with col2:
                    # Summary report
                    summary_data = {
                        'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'model1_type': st.session_state.swimming_model1_type,
                        'model1_accuracy': st.session_state.swimming_model1['accuracy'],
                        'model2_type': st.session_state.swimming_model2_type,
                        'model2_accuracy': st.session_state.swimming_model2['accuracy'],
                        'total_segments': total_segments,
                        'agreement_rate': agreement_rate,
                        'model1_avg_conf': avg_conf1,
                        'model2_avg_conf': avg_conf2,
                        'model1_low_conf': model1_low_conf,
                        'model2_low_conf': model2_low_conf,
                        'disagreements': total_segments - agreement_count,
                        'segment_duration': segment_duration,
                        'confidence_threshold': confidence_threshold
                    }
                    
                    summary_df = pd.DataFrame([summary_data])
                    summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📋 Download Summary",
                        data=summary_csv,
                        file_name="swimming_comparison_summary.csv",
                        mime="text/csv",
                        help="High-level comparison statistics"
                    )
                
                with col3:
                    # Visualization export info
                    st.info("""
                    **Visualizations:**
                    - Right-click any chart to save as PNG
                    - Use browser print for PDF reports
                    - All data available in CSV downloads
                    """)
                
                st.success("🎉 Swimming model comparison complete! Download results above.")
            
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ==========================
    # FOOTER
    # ==========================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>🏊‍♂️ Swimming Stroke Recognition Model Comparison Tool | Academic Project</p>
        <p>📊 Compare KNN vs Random Forest predictions on new swimming data (16 features)</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================
# FOR STANDALONE EXECUTION (optional)
# ==========================
if __name__ == "__main__":
    run_swimming_app()