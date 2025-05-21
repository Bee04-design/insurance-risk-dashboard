from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import shap
import plotly.express as px
import folium
from folium.plugins import HeatMap
import geopandas
from shapely.geometry import Point
from streamlit_folium import st_folium
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import logging
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy.stats import ks_2samp
from datetime import datetime
from sklearn.utils import resample
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Setup Logging with Version Control
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v1.0"
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-20 12:10:00"  # Updated to current time

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="Insurance Risk Dashboard", page_icon="📊", layout="wide")

# Title and Version Info
st.title("Insurance Risk Analytics Dashboard")
st.markdown(f"_Prototype v0.4.6 | Model: {MODEL_VERSION} | Dataset: {DATASET_VERSION} | Last Trained: {MODEL_LAST_TRAINED}_")

# Sidebar for File Upload
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file ")

if uploaded_file is None:
    st.info("Upload a file through config", icon="ℹ️")
    st.stop()

    preview_option = st.selectbox("Preview Dataset", ["No Preview", "Head", "Tail", "Sample (10 rows)"])
    if preview_option != "No Preview" and uploaded_file is not None:
        df_preview = load_data(uploaded_file)
        if preview_option == "Head":
            st.write(df_preview.head())
        elif preview_option == "Tail":
            st.write(df_preview.tail())
        elif preview_option == "Sample (10 rows)":
            st.write(df_preview.sample(10))
# Load Data
@st.cache_data
def load_data(path):
    logger.info("Loading data...")
    df = pd.read_csv(path)
    logger.info("Data loaded successfully")
    return df

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Dataset loading failed: {str(e)}")
    logger.error(f"Dataset loading failed: {str(e)}")
    st.stop()

    # --- Target Selection ---
target_col = st.selectbox("Select the target column", df.columns)
def preprocess_data(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Combine for consistent row drops (e.g., due to NaNs or infs)
    combined = pd.concat([X, y], axis=1)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    
    # Separate again
    y = combined[target_col]
    X = combined.drop(columns=[target_col])

    # Define preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

def train_random_forest_model(X, y):
    # Check if stratified split is possible
    class_counts = y.value_counts()
    if class_counts.min() >= 2:
        stratify_option = y
    else:
        st.warning("Stratified split disabled: Some classes have < 2 samples.")
        stratify_option = None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify_option, random_state=42
    )

    # Apply ADASYN oversampling
    adasyn = ADASYN(random_state=42)
    X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

    # Feature selection using RandomForest
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_model.fit(X_train_balanced, y_train_balanced)
    selector = SelectFromModel(selector_model, prefit=True)

    X_train_sel = selector.transform(X_train_balanced)
    X_test_sel = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()]

    # Hyperparameter tuning with GridSearch
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1_weighted'
    )
    grid_search.fit(X_train_sel, y_train_balanced)

    # Predict and evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_sel)
    report = classification_report(y_test, y_pred, output_dict=True)

    return best_model, selected_features, X_test_sel, y_test, report
    # --- Run Preprocessing and Modeling ---
X_processed, y= preprocess_data(df.copy(), target_col)



best_model, selected_features, X_test_sel, y_test, report = train_random_forest_model(X, y)

    # --- ROC Curve ---
y_prob = best_model.predict_proba(X_test_sel)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

    # --- Clustering ---
silhouette = []
range_n_clusters = range(2, 10)
X_segment = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_segment)
        silhouette.append(silhouette_score(X_segment, labels))
optimal_clusters = range_n_clusters[np.argmax(silhouette)]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_cleaned['customer_segment'] = kmeans.fit_predict(X_segment).astype(str)

    # --- Metrics Display ---
st.subheader("Key Metrics")
cols = st.columns(4)
metrics = [
        {"label": "Total Records", "value": len(df)},
        {"label": "Model AUC", "value": f"{roc_auc:.2f}"},
        {"label": "Missing Values", "value": df.isnull().sum().sum()},
        {"label": "Selected Features", "value": len(selected_features)}
    ]
for col, metric in zip(cols, metrics):
       col.metric(label=metric["label"], value=metric["value"])

        # --- Plot ROC Curve ---
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

        # --- Classification Report ---
st.subheader("Classification Report")
st.json(report)

st.success("Model trained and results displayed successfully!")

# Map Functions with Segmentations




def init_map(center=(-26.5, 31.5), zoom_start=7, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)

def create_point_map(df):
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].apply(pd.to_numeric, errors='coerce')
    df['coordinates'] = df[['Latitude', 'Longitude']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = geopandas.GeoDataFrame(df, geometry='coordinates')
    df = df.dropna(subset=['Latitude', 'Longitude', 'coordinates'])
    return df

def plot_from_df(df, folium_map, selected_risk_levels, selected_regions, selected_segments):
    region_coords = {
        'Lubombo': (-26.3, 31.8), 'Hhohho': (-26.0, 31.1),
        'Manzini': (-26.5, 31.4), 'Shiselweni': (-27.0, 31.3)
    }
    risk_by_region = df.groupby('location')['claim_risk'].mean().reset_index()
    risk_by_region = risk_by_region[risk_by_region['location'].isin(region_coords.keys())]
    risk_by_region['Latitude'] = risk_by_region['location'].map(lambda x: region_coords[x][0])
    risk_by_region['Longitude'] = risk_by_region['location'].map(lambda x: region_coords[x][1])
    
    # Choropleth style
    folium.Choropleth(
        geo_data='eswatini_regions.geojson',  # Add a GeoJSON file for Eswatini regions
        name='choropleth',
        data=risk_by_region,
        columns=['location', 'claim_risk'],
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Claim Risk'
    ).add_to(folium_map)

    # Gauge for overall risk
    folium.plugins.MiniMap().add_to(folium_map)  # Optional mini-map
    return folium_map
    # Aggregate risk by region and segment
    risk_by_region_segment = df.groupby(['location', 'customer_segment'])['claim_risk'].mean().reset_index()
    risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(region_coords.keys())]
    risk_by_region_segment['Latitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][0])
    risk_by_region_segment['Longitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][1])
    
    # Ensure sufficient unique values for qcut
    if risk_by_region_segment['claim_risk'].nunique() > 1:
        # Calculate quantiles manually to avoid bin edge issues
        quantiles = risk_by_region_segment['claim_risk'].quantile([0, 0.33, 0.66, 1]).values
        risk_by_region_segment['risk_level'] = pd.cut(risk_by_region_segment['claim_risk'], bins=quantiles, labels=['Low', 'Medium', 'High'], include_lowest=True)
    else:
        risk_by_region_segment['risk_level'] = 'Low'  # Default if not enough variance

    # Apply filters
    if selected_risk_levels:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['risk_level'].isin(selected_risk_levels)]
    if selected_regions:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(selected_regions)]
    if selected_segments:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['customer_segment'].isin(selected_segments)]

    # Plot markers with segment-specific styling
    segment_styles = {
        '0': {'radius': 10, 'color': '#1f77b4'},
        '1': {'radius': 12, 'color': '#ff7f0e'},
        '2': {'radius': 14, 'color': '#2ca02c'},
        '3': {'radius': 16, 'color': '#d62728'}
    }
    for i, row in risk_by_region_segment.iterrows():
        style = segment_styles.get(row['customer_segment'], {'radius': 10, 'color': '#1f77b4'})
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=style['radius'],
            color=style['color'],
            fill=True,
            fill_color=style['color'],
            fill_opacity=0.7,
            tooltip=f"{row['location']} (Segment {row['customer_segment']}): {row['risk_level']} Risk ({row['claim_risk']*100:.1f}%)"
        ).add_to(folium_map)

    # Add heatmap for high-risk claims
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows() if row['claim_risk'] == 1]
    HeatMap(heat_data, radius=15).add_to(folium_map)
    return folium_map
@st.cache_data
def load_map(df, selected_risk_levels, selected_regions, selected_segments):
    m = init_map()
    m = plot_from_df(df, m, selected_risk_levels, selected_regions, selected_segments)
    return m

# Section 1: Prediction
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.header("Predict Claim Risk")
    input_data = {}
    for col in df.columns:
        if col in ['claim_amount_SZL', 'claim_risk'] or col in categorical_cols or col in date_cols:
            continue
        try:
            if df[col].dtype in ['int64', 'float64']:
                input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col}", df[col].unique())
        except Exception as e:
            st.warning(f"Error with {col}: {str(e)}. Using default value.")
            input_data[col] = 0 if df[col].dtype in ['int64', 'float64'] else df[col].mode()[0]

    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", df[col].unique())

    for col in date_cols:
        if col in df.columns:
            continue
        input_data[f'{col}_year'] = st.slider(f"{col} Year", 2000, 2025, 2020)
        input_data[f'{col}_month'] = st.slider(f"{col} Month", 1, 12, 6)
        input_data[f'{col}_day'] = st.slider(f"{col} Day", 1, 31, 15)

    if st.button("Predict"):
        logger.info("Predict button clicked")
        try:
            input_df = pd.DataFrame([input_data])
            expected_features = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else X_test.columns
            input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
            for col in expected_features:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0
            input_df_encoded = input_df_encoded[expected_features]
            pred = rf.predict(input_df_encoded)[0]
            prob = rf.predict_proba(input_df_encoded)[0][1]
            with col2:
                st.markdown(f"**Prediction**: {'High Risk' if pred == 1 else 'Low Risk'}")
            with col3:
                st.metric("Probability (High Risk)", f"{prob*100:.1f}%")
                st.progress(prob)
            logger.info(f"Prediction: {pred}, Probability: {prob}")
            pred_log = pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'prediction': ['High Risk' if pred == 1 else 'Low Risk'],
                'probability_high_risk': [prob]
            })
            log_file = os.path.join(save_dir, 'prediction_log.csv')
            if os.path.exists(log_file):
                pred_log.to_csv(log_file, mode='a', header=False, index=False)
            else:
                pred_log.to_csv(log_file, index=False)
            logger.info("Prediction saved to prediction_log.csv")

            # Drift Detection (Moved here)
            from scipy.stats import ks_2samp
            drift_feature = 'claim_amount_SZL'
            if drift_feature in df.columns:
                original_dist = df[drift_feature].values
                current_dist = pd.DataFrame([input_data]).get(drift_feature, [df[drift_feature].mean()])[0]
                stat, p_value = ks_2samp(original_dist, np.array([current_dist] * len(original_dist)))
                if p_value < 0.05:
                    st.warning(f"Data drift detected in {drift_feature} (p-value: {p_value:.4f}). Consider retraining the model.")
                    logger.warning(f"Data drift detected in {drift_feature} (p-value: {p_value:.4f})")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            logger.error(f"Prediction failed: {str(e)}")

# Section 2: Model Performance and Risk Trends
with st.expander("Model Performance", expanded=True):
    col4, col5 = st.columns(2)
    with col4:
        st.header("Confusion Matrix")
        try:
            y_pred = rf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig_cm)
        except Exception as e:
            st.error(f"Performance plotting failed: {str(e)}")
    with col5:
        st.header("ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
            fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'FPR', 'y': 'TPR'})
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'))
            fig_roc.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_roc)
            st.text(f"Recall for High Risk: {recall_class_1:.2f}")
        except Exception as e:
            st.error(f"ROC plotting failed: {str(e)}")
            
    with col6:
       st.header("Risk Trend Over Time")
    try:
        log_file = os.path.join(save_dir, 'prediction_log.csv')
        if os.path.exists(log_file):
            pred_log = pd.read_csv(log_file)
            pred_log['timestamp'] = pd.to_datetime(pred_log['timestamp'])
            fig_trend = px.line(pred_log, x='timestamp', y='probability_high_risk', title="High Risk Probability Trend", markers=True)
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
            logger.info("Risk trend plot rendered")
        else:
            st.info("No prediction history available to display trends.")
    except Exception as e:
        st.error(f"Risk trend plotting failed: {str(e)}")
        logger.error(f"Risk trend plotting failed: {str(e)}")

# Section 3: Feature Importance and Risk Distributions
col7, col8, col9 = st.columns([1, 1, 1])
with col7:
    st.header("Risk Driver Insights (SHAP)")
    with st.spinner("Computing SHAP values..."):
        try:
            explainer = shap.TreeExplainer(rf)
            sample_data = X_test.sample(50, random_state=42)
            sample_encoded = pd.get_dummies(sample_data, columns=[col for col in categorical_cols if col in sample_data.columns], drop_first=False)
            expected_features = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else X_test.columns
            for col in expected_features:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            sample_encoded = sample_encoded[expected_features].values
            shap_values = explainer.shap_values(sample_encoded)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.array(shap_values).reshape(-1, len(expected_features))
            st.subheader("Features Used in SHAP Analysis")
            st.write(list(expected_features))
            fig_shap = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_encoded, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
            if plt.gcf().axes:
                plt.title('Top Features for High Risk')
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.savefig('shap_plot.png')
            shap_df = pd.DataFrame({'Feature': expected_features, 'SHAP Value': np.abs(shap_values).mean(axis=0)}).sort_values(by='SHAP Value', ascending=False).head(5)
            st.session_state['shap_df'] = shap_df
            logger.info("SHAP plot rendered")
        except Exception as e:
            st.error(f"SHAP plot failed: {str(e)}")
            logger.error(f"SHAP plot failed: {str(e)}")

with col8:
   col8, col9 = st.columns(2)
with col8:
    st.header("Risk by Location")
    fig_loc = px.bar(risk_by_location, x='location', y='claim_risk', title="Average Risk by Location (%)",
                     color='claim_risk', color_continuous_scale='Blues', height=300)
    fig_loc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_loc)
with col9:
    st.header("Risk by Claim Type")
    fig_claim = px.bar(risk_by_claim_type, x='claim_type', y='claim_risk', title="Average Risk by Claim Type (%)",
                       color='claim_risk', color_continuous_scale='Blues', height=300)
    fig_claim.update_layout(template="plotly_dark")
    st.plotly_chart(fig_claim)
# Section 4: Segmentation Drill-down
col10, col11 = st.columns([1, 1])
with col10:
    st.header("Customer Segment Drill-down")
    segment = st.selectbox("Select Customer Segment", df['customer_segment'].unique())
    segment_data = df[df['customer_segment'] == segment]
    try:
        # Check for variance in claim_amount_SZL to avoid binning issues
        if segment_data['claim_amount_SZL'].nunique() > 1:
            fig_segment_trend = px.histogram(segment_data, x='claim_amount_SZL', color='claim_risk', title=f"Claim Amount Distribution in {segment}", nbins=20)
        else:
            raise ValueError("Not enough variance in data for histogram")
        fig_segment_trend.update_layout(height=300)
        st.plotly_chart(fig_segment_trend, use_container_width=True)
    except ValueError as e:
        st.warning(f"Histogram failed: {str(e)}. Using bar plot instead.")
        fig_segment_trend = px.bar(segment_data.groupby('claim_risk').size().reset_index(name='Count'), x='claim_risk', y='Count', title=f"Risk Distribution in {segment}")
        fig_segment_trend.update_layout(height=300)
        st.plotly_chart(fig_segment_trend, use_container_width=True)
    logger.info(f"Segment trend plot rendered for {segment}")

with col11:
    st.header("Top Features for Segment")
    try:
        segment_encoded = pd.get_dummies(segment_data.drop(columns=['claim_amount_SZL', 'claim_risk']), columns=categorical_cols, drop_first=False)
        for col in expected_features:
            if col not in segment_encoded.columns:
                segment_encoded[col] = 0
        segment_encoded = segment_encoded[expected_features].values
        shap_values_segment = explainer.shap_values(segment_encoded)
        if isinstance(shap_values_segment, list):
            shap_values_segment = shap_values_segment[1]
        shap_values_segment = np.array(shap_values_segment).reshape(-1, len(expected_features))
        fig_shap_segment = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_segment, segment_encoded, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
        if plt.gcf().axes:
            plt.title(f'Top Features for {segment}')
            plt.tight_layout()
            st.pyplot(fig_shap_segment)
        logger.info(f"SHAP plot for segment {segment} rendered")
    except Exception as e:
        st.error(f"SHAP plot for segment failed: {str(e)}")
        logger.error(f"SHAP plot for segment failed: {str(e)}")

# Section 5: Interactive Eswatini Risk Map with Segmentations
col12 = st.columns([3])[0]
with col12:
    st.header("Interactive Eswatini Risk Map with Segmentations")
    risk_levels = st.multiselect("Filter by Risk Level", ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
    regions = st.multiselect("Filter by Region", ['Lubombo', 'Hhohho', 'Manzini', 'Shiselweni'], default=['Lubombo', 'Hhohho', 'Manzini', 'Shiselweni'])
    customer_segments = st.multiselect("Filter by Customer Segment", df['customer_segment'].unique(), default=df['customer_segment'].unique())
    try:
        m = load_map(df, risk_levels, regions, customer_segments)
        map_data = st_folium(m, height=500, width=1000, key="eswatini_map")
        selected_region = map_data.get('last_object_clicked_tooltip', '').split(':')[0].strip() if map_data.get('last_object_clicked_tooltip') else None

        if selected_region:
            st.subheader(f"Risk Analysis for {selected_region}")
            region_data = df[df['location'] == selected_region]
            if not region_data.empty:
                try:
                    if region_data['claim_amount_SZL'].nunique() > 1:
                        fig_region_dist = px.histogram(region_data, x='claim_amount_SZL', color='claim_risk', title=f"Claim Amount Distribution in {selected_region}", nbins=20)
                    else:
                        raise ValueError("Not enough variance in data for histogram")
                    fig_region_dist.update_layout(height=300)
                    st.plotly_chart(fig_region_dist, use_container_width=True)
                except ValueError as e:
                    st.warning(f"Histogram failed: {str(e)}. Using bar plot instead.")
                    fig_region_dist = px.bar(region_data.groupby('claim_risk').size().reset_index(name='Count'), x='claim_risk', y='Count', title=f"Risk Distribution in {selected_region}")
                    fig_region_dist.update_layout(height=300)
                    st.plotly_chart(fig_region_dist, use_container_width=True)
                sample_region = region_data.sample(min(20, len(region_data)), random_state=42)
                sample_encoded_region = pd.get_dummies(sample_region.drop(columns=['claim_amount_SZL', 'claim_risk']), columns=categorical_cols, drop_first=False)
                for col in expected_features:
                    if col not in sample_encoded_region.columns:
                        sample_encoded_region[col] = 0
                sample_encoded_region = sample_encoded_region[expected_features].values
                shap_values_region = explainer.shap_values(sample_encoded_region)
                if isinstance(shap_values_region, list):
                    shap_values_region = shap_values_region[1]
                shap_values_region = np.array(shap_values_region).reshape(-1, len(expected_features))
                fig_shap_region = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_region, sample_encoded_region, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
                if plt.gcf().axes:
                    plt.title(f'Top Features for High Risk in {selected_region}')
                    plt.tight_layout()
                    st.pyplot(fig_shap_region)
            else:
                st.warning(f"No data available for {selected_region}.")
        logger.info("Interactive map and region analysis rendered")
    except Exception as e:
        st.error(f"Map rendering or analysis failed: {str(e)}")
        logger.error(f"Map rendering or analysis failed: {str(e)}")

# Section 6: Downloadable Reports and Data
def generate_pdf():
    pdf_file = "insurance_risk_report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    y_position = 750
    c.drawString(100, y_position, "Insurance Risk Report")
    y_position -= 20
    c.drawString(100, y_position, f"Generated on: {datetime.now()}")
    y_position -= 20
    c.drawString(100, y_position, f"Model AUC: {roc_auc:.2f}")
    y_position -= 20
    c.drawString(100, y_position, f"Recall for High Risk: {recall_class_1:.2f}")
    y_position -= 30
    c.drawString(100, y_position, "Top Features (SHAP):")
    y_position -= 20
    if 'shap_df' in st.session_state:
        for i, row in st.session_state['shap_df'].iterrows():
            c.drawString(100, y_position, f"{row['Feature']}: {row['SHAP Value']:.4f}")
            y_position -= 15
    c.save()
    return pdf_file
# Example usage in your app
if st.button("Download PDF Report"):
    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name="report.pdf")
# Notes
st.markdown("**Note**: Ensure the dataset is available. Risk map uses claim risk data to highlight high-risk areas.", unsafe_allow_html=True)
st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
