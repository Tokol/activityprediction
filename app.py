import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
import pickle
import tempfile
import os
from datetime import datetime
import matplotlib  # Add this import

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
    st.stop()

# Import shared modules
try:
    from features import compute_magnitude
except ImportError as e:
    st.error(f"❌ Error importing shared modules: {str(e)}")
    st.stop()

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Predicting Activity",
    layout="wide",
    page_icon="assets/favicon.ico"
)

st.title("🎯 Activity Recognition: Model Comparison Predictor")
st.markdown("""
**Academic Tool:** Load two trained models, compare predictions, and analyze algorithm performance.
Upload KNN and Random Forest models to see where they agree/disagree on new data.
""")

# ==========================
# SESSION STATE
# ==========================
if 'model1' not in st.session_state:
    st.session_state.model1 = None
    st.session_state.model1_type = None

if 'model2' not in st.session_state:
    st.session_state.model2 = None
    st.session_state.model2_type = None

if 'new_data' not in st.session_state:
    st.session_state.new_data = None

if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

if 'features_df' not in st.session_state:
    st.session_state.features_df = None

# ==========================
# SIDEBAR - MODEL UPLOAD
# ==========================
with st.sidebar:
    st.image("assets/logo.png", width=150)  # ← Add this line
    st.markdown("---")  # Optional separator

    st.header("🔧 Model Configuration")

    # MODEL 1 UPLOAD
    st.subheader("1. Upload Model 1")
    model1_file = st.file_uploader(
        "📤 Upload first model (.pkl)",
        type=["pkl"],
        key="model1_uploader",
        help="Upload KNN or Random Forest model"
    )

    if model1_file is not None and st.session_state.model1 is None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(model1_file.getvalue())
                tmp_path = tmp_file.name

            with open(tmp_path, 'rb') as f:
                loaded_data = pickle.load(f)

            os.unlink(tmp_path)

            st.session_state.model1 = loaded_data
            st.session_state.model1_type = "KNN" if 'scaler' in loaded_data else "Random Forest"

            st.success(f"✅ Model 1 Loaded ({st.session_state.model1_type})")

            with st.expander("📋 Model 1 Details", expanded=True):
                st.write(f"**Type:** {st.session_state.model1_type}")
                st.write(f"**Accuracy:** {loaded_data['accuracy']:.3f}")
                st.write(f"**Features:** {len(loaded_data['feature_names'])} features")
                st.write(f"**Classes:** {', '.join(loaded_data['classes'])}")
                if 'training_date' in loaded_data:
                    st.write(f"**Trained:** {loaded_data['training_date']}")
                if st.session_state.model1_type == "KNN":
                    st.write(f"**Optimal k:** {loaded_data.get('optimal_k', 'N/A')}")
                else:
                    st.write(f"**Trees:** {loaded_data.get('n_estimators', 'N/A')}")

        except Exception as e:
            st.error(f"❌ Error loading Model 1: {str(e)}")

    # MODEL 2 UPLOAD
    st.subheader("2. Upload Model 2")
    model2_file = st.file_uploader(
        "📤 Upload second model (.pkl)",
        type=["pkl"],
        key="model2_uploader",
        help="Upload a different model for comparison"
    )

    if model2_file is not None and st.session_state.model2 is None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(model2_file.getvalue())
                tmp_path = tmp_file.name

            with open(tmp_path, 'rb') as f:
                loaded_data = pickle.load(f)

            os.unlink(tmp_path)

            st.session_state.model2 = loaded_data
            st.session_state.model2_type = "KNN" if 'scaler' in loaded_data else "Random Forest"

            st.success(f"✅ Model 2 Loaded ({st.session_state.model2_type})")

            with st.expander("📋 Model 2 Details", expanded=True):
                st.write(f"**Type:** {st.session_state.model2_type}")
                st.write(f"**Accuracy:** {loaded_data['accuracy']:.3f}")
                st.write(f"**Features:** {len(loaded_data['feature_names'])} features")
                st.write(f"**Classes:** {', '.join(loaded_data['classes'])}")
                if 'training_date' in loaded_data:
                    st.write(f"**Trained:** {loaded_data['training_date']}")
                if st.session_state.model2_type == "KNN":
                    st.write(f"**Optimal k:** {loaded_data.get('optimal_k', 'N/A')}")
                else:
                    st.write(f"**Trees:** {loaded_data.get('n_estimators', 'N/A')}")

        except Exception as e:
            st.error(f"❌ Error loading Model 2: {str(e)}")

    st.markdown("---")

    # DATA PROCESSING SETTINGS
    st.subheader("3. Processing Settings")

    segment_duration = st.slider(
        "Segment Duration (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Window size for feature extraction"
    )

    confidence_threshold = st.slider(
        "Low Confidence Threshold (%)",
        min_value=50,
        max_value=90,
        value=70,
        help="Flag predictions below this confidence"
    ) / 100

    # Reset button
    if st.button("🔄 Reset All", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Model comparison info
    if st.session_state.model1 is not None and st.session_state.model2 is not None:
        st.markdown("---")
        st.subheader("📊 Model Comparison Ready")

        model1_acc = st.session_state.model1['accuracy']
        model2_acc = st.session_state.model2['accuracy']

        if model1_acc > model2_acc:
            st.info(f"**Model 1 has higher accuracy** (+{(model1_acc - model2_acc) * 100:.1f}%)")
        elif model2_acc > model1_acc:
            st.info(f"**Model 2 has higher accuracy** (+{(model2_acc - model1_acc) * 100:.1f}%)")
        else:
            st.info("**Models have equal accuracy**")

        if st.session_state.model1_type == st.session_state.model2_type:
            st.warning("⚠️ Both models are same type. Consider uploading different algorithms.")

# ==========================
# MAIN INTERFACE
# ==========================
# Check if models are loaded
if st.session_state.model1 is None or st.session_state.model2 is None:
    # Show instructions if models not loaded
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 How to Use:

        1. **Upload Two Models** (sidebar):
           - Model 1: KNN or Random Forest
           - Model 2: Different algorithm recommended

        2. **Upload New CSV Data**:
           - Same format as training: Time, X, Y, Z columns

        3. **Compare Predictions**:
           - See where models agree/disagree
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
    with st.expander("💡 Example Model Types for Comparison", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🤖 KNN Model:**
            - 4 motion features only
            - Distance-based algorithm
            - Fast predictions
            - Sensitive to feature scaling
            """)
        with col2:
            st.markdown("""
            **🌲 Random Forest Model:**
            - 9 features (4 motion + 5 shape)
            - Ensemble of decision trees
            - Handles complex patterns
            - Provides feature importance
            """)

    st.info("👈 **Start by uploading two models in the sidebar!**")
    st.stop()

# Models are loaded - show main interface
st.success(f"✅ Ready: Model 1 ({st.session_state.model1_type}) vs Model 2 ({st.session_state.model2_type})")

# ==========================
# DATA UPLOAD SECTION
# ==========================
st.header("📤 Step 3: Upload New Data")

new_csv = st.file_uploader(
    "Upload new CSV file for prediction",
    type=["csv"],
    help="CSV with columns: Time (s), X (m/s^2), Y (m/s^2), Z (m/s^2)"
)

if new_csv is None:
    st.info("📝 Upload a CSV file to start prediction comparison")
    st.stop()

try:
    # Load and process new data
    new_df = pd.read_csv(new_csv)

    # Validate columns
    required_cols = ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
    if not all(col in new_df.columns for col in required_cols):
        st.error("❌ CSV must contain required columns")
        st.stop()

    # Compute magnitude
    new_df = compute_magnitude(new_df)
    st.session_state.new_data = new_df

    # Show data preview
    with st.expander("📄 Data Preview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(new_df))
            st.metric("Duration", f"{new_df['Time (s)'].max() - new_df['Time (s)'].min():.1f}s")
        with col2:
            st.metric("Time Range", f"{new_df['Time (s)'].min():.1f}s to {new_df['Time (s)'].max():.1f}s")
        with col3:
            if len(new_df) > 1:
                sampling_rate = 1 / np.mean(np.diff(new_df['Time (s)']))
                st.metric("Sampling Rate", f"{sampling_rate:.1f} Hz")
            else:
                st.metric("Sampling Rate", "N/A")

        st.dataframe(new_df.head(), use_container_width=True)

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# ==========================
# PROCESSING AND PREDICTION
# ==========================
if st.button("⚡ Compare Model Predictions", type="primary", use_container_width=True):
    with st.spinner("Extracting features and comparing predictions..."):
        try:
            # Extract features from all data
            features = []
            start_time = new_df["Time (s)"].min()
            end_time = new_df["Time (s)"].max()

            for seg_start in np.arange(start_time, end_time, segment_duration):
                seg_end = seg_start + segment_duration

                seg_df = new_df[
                    (new_df["Time (s)"] >= seg_start) &
                    (new_df["Time (s)"] < seg_end)
                    ]

                if len(seg_df) < 10:  # Minimum samples for meaningful features
                    continue

                mag = seg_df["magnitude"]

                # Extract all possible features
                feature_dict = {
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "duration": seg_end - seg_start,
                    "sample_count": len(seg_df),

                    # Motion features (for KNN)
                    "mean_magnitude": mag.mean(),
                    "rms_magnitude": np.sqrt(np.mean(mag ** 2)),
                    "std_magnitude": mag.std(),
                    "p2p_magnitude": mag.max() - mag.min(),

                    # Shape features (for Random Forest)
                    "kurtosis_magnitude": mag.kurt(),
                    "skewness_magnitude": mag.skew(),
                    "median_magnitude": mag.median(),
                    "iqr_magnitude": mag.quantile(0.75) - mag.quantile(0.25),
                }

                # Peak count
                peaks, _ = find_peaks(mag, height=np.mean(mag))
                feature_dict["peak_count"] = len(peaks)

                features.append(feature_dict)

            features_df = pd.DataFrame(features)

            if features_df.empty:
                st.warning("⚠️ No complete segments found in the data.")
                st.stop()

            st.session_state.features_df = features_df

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
                model1_data = st.session_state.model1
                feature_names1 = model1_data['feature_names']

                # Select features for model 1
                X1 = row[feature_names1].values.reshape(1, -1)

                # Scale if KNN
                if st.session_state.model1_type == "KNN":
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
                model2_data = st.session_state.model2
                feature_names2 = model2_data['feature_names']

                # Select features for model 2
                X2 = row[feature_names2].values.reshape(1, -1)

                # Scale if KNN
                if st.session_state.model2_type == "KNN":
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
            st.session_state.predictions_df = predictions_df

            # ==========================
            # DYNAMIC COLOR MAPPING FOR ACTIVITIES
            # ==========================
            # Get all unique activities from both models
            all_activities = set()
            if 'classes' in st.session_state.model1:
                all_activities.update(st.session_state.model1['classes'])
            if 'classes' in st.session_state.model2:
                all_activities.update(st.session_state.model2['classes'])

            # Also get activities from predictions
            all_activities.update(predictions_df['model1_pred'].unique())
            all_activities.update(predictions_df['model2_pred'].unique())

            all_activities = list(all_activities)

            # Create dynamic color mapping
            matplotlib_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

            plotly_colors = ['blue', 'orange', 'green', 'red', 'purple',
                             'brown', 'pink', 'gray', 'olive', 'cyan']

            # Create color mappings for both matplotlib and plotly
            matplotlib_activity_colors = {}
            plotly_activity_colors = {}

            for i, activity in enumerate(all_activities):
                matplotlib_activity_colors[activity] = matplotlib_colors[i % len(matplotlib_colors)]
                plotly_activity_colors[activity] = plotly_colors[i % len(plotly_colors)]

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
                for activity in predictions_df['model1_pred'].unique():
                    activity_data = predictions_df[predictions_df['model1_pred'] == activity]
                    if not activity_data.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=activity_data['start_time'],
                            y=['Model 1'] * len(activity_data),
                            mode='markers',
                            name=f'Model 1: {activity}',
                            marker=dict(
                                size=10,
                                color=plotly_activity_colors.get(activity, 'gray'),
                                symbol='circle',
                                opacity=0.7
                            ),
                            hovertemplate='<b>Model 1</b><br>' +
                                          'Activity: %{text}<br>' +
                                          'Time: %{x:.1f}s<br>' +
                                          'Confidence: %{customdata:.1%}<extra></extra>',
                            text=[activity] * len(activity_data),
                            customdata=activity_data['model1_conf']
                        ))

                # Add Model 2 predictions
                for activity in predictions_df['model2_pred'].unique():
                    activity_data = predictions_df[predictions_df['model2_pred'] == activity]
                    if not activity_data.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=activity_data['start_time'],
                            y=['Model 2'] * len(activity_data),
                            mode='markers',
                            name=f'Model 2: {activity}',
                            marker=dict(
                                size=10,
                                color=plotly_activity_colors.get(activity, 'gray'),
                                symbol='square',
                                opacity=0.7
                            ),
                            hovertemplate='<b>Model 2</b><br>' +
                                          'Activity: %{text}<br>' +
                                          'Time: %{x:.1f}s<br>' +
                                          'Confidence: %{customdata:.1%}<extra></extra>',
                            text=[activity] * len(activity_data),
                            customdata=activity_data['model2_conf']
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
                    title='Model Predictions Timeline',
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

                    # Select key features for analysis
                    key_features = ['mean_magnitude', 'rms_magnitude', 'std_magnitude', 'p2p_magnitude']

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
                        title='Feature Values for Disagreement Segments',
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
            st.subheader("📈 Raw Signal with Model Predictions")

            # Create figure
            fig_signal, ax = plt.subplots(figsize=(12, 4))

            # Plot raw signal
            ax.plot(new_df["Time (s)"], new_df["magnitude"],
                    color="black", alpha=0.6, linewidth=0.8, label="Raw Signal")

            # Add colored regions for Model 1 predictions using dynamic colors
            for _, row in predictions_df.iterrows():
                activity = row['model1_pred']
                color = matplotlib_activity_colors.get(activity, 'gray')

                # Different opacity for disagreements
                alpha = 0.15 if row['agree'] else 0.3

                ax.axvspan(row['start_time'], row['end_time'],
                           alpha=alpha, color=color)

            # Custom legend
            from matplotlib.patches import Patch

            legend_elements = []
            # Only show activities that actually appear in predictions
            predicted_activities = predictions_df['model1_pred'].unique()
            for activity in predicted_activities:
                color = matplotlib_activity_colors.get(activity, 'gray')
                display_name = activity.replace('_', ' ').title()
                legend_elements.append(
                    Patch(facecolor=color, alpha=0.2,
                          label=f"{display_name} (Model 1)")
                )

            # Add disagreement markers
            disagreements = predictions_df[~predictions_df['agree']]
            if not disagreements.empty:
                for _, row in disagreements.iterrows():
                    mid_time = (row['start_time'] + row['end_time']) / 2
                    signal_val = new_df.loc[
                        new_df["Time (s)"].sub(mid_time).abs().idxmin(),
                        "magnitude"
                    ]
                    ax.plot(mid_time, signal_val, 'rx', markersize=10,
                            label='Disagreement' if _ == disagreements.index[0] else "")

            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper left',
                          bbox_to_anchor=(1.01, 1))

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Magnitude (m/s²)")
            ax.grid(True, alpha=0.3)
            ax.set_title("Acceleration Signal with Model 1 Predictions (Red X = Disagreement)")

            st.pyplot(fig_signal)

            # ==========================
            # PERFORMANCE METRICS BY ACTIVITY
            # ==========================
            st.subheader("🎯 Performance by Activity")

            # Calculate agreement by activity
            activity_agreement = []
            for activity in predictions_df['model1_pred'].unique():
                activity_data = predictions_df[predictions_df['model1_pred'] == activity]
                if len(activity_data) > 0:
                    agreement_rate_act = activity_data['agree'].mean()
                    avg_conf1_act = activity_data['model1_conf'].mean()
                    avg_conf2_act = activity_data['model2_conf'].mean()

                    activity_agreement.append({
                        'Activity': activity.replace('_', ' ').title(),
                        'Segments': len(activity_data),
                        'Agreement Rate': agreement_rate_act,
                        'Model 1 Avg Conf': avg_conf1_act,
                        'Model 2 Avg Conf': avg_conf2_act,
                        'Conf Difference': avg_conf1_act - avg_conf2_act
                    })

            agreement_df = pd.DataFrame(activity_agreement)

            if not agreement_df.empty:
                # Display table
                st.dataframe(agreement_df, use_container_width=True)

                # Agreement rate by activity bar chart
                fig_agreement = go.Figure(data=[
                    go.Bar(
                        x=agreement_df['Activity'],
                        y=agreement_df['Agreement Rate'],
                        text=agreement_df['Agreement Rate'].apply(lambda x: f"{x:.1%}"),
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])

                fig_agreement.update_layout(
                    title='Agreement Rate by Activity (Model 1 Perspective)',
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

            # Create download dataframe without merge to avoid column conflicts
            download_data = predictions_df.copy()

            # Add feature columns (excluding the ones already in predictions_df)
            feature_cols_to_add = [
                'mean_magnitude', 'rms_magnitude', 'std_magnitude', 'p2p_magnitude',
                'kurtosis_magnitude', 'skewness_magnitude', 'median_magnitude',
                'iqr_magnitude', 'peak_count'
            ]

            for col in feature_cols_to_add:
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
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Complete dataset with all predictions and features"
                )

            with col2:
                # Summary report
                summary_data = {
                    'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model1_type': st.session_state.model1_type,
                    'model1_accuracy': st.session_state.model1['accuracy'],
                    'model2_type': st.session_state.model2_type,
                    'model2_accuracy': st.session_state.model2['accuracy'],
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
                    file_name="comparison_summary.csv",
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

            st.success("🎉 Model comparison complete! Download results above.")

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
    <p>🎯 Activity Recognition Model Comparison Tool | Academic Project</p>
    <p>📊 Compare KNN vs Random Forest predictions on new accelerometer data</p>
</div>
""", unsafe_allow_html=True)