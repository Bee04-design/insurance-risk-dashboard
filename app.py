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
from streamlit_folium import folium_static
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
import logging
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy.stats import ks_2samp
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Setup Logging with Version Control
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v1.0"
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-22 19:55:00"

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="Insurance Risk Dashboard", page_icon="📊", layout="wide")

# Title and Version Info
st.title("Insurance Risk Streamlit Dashboard")
st.markdown(f"_Prototype v0.4.6 | Model: {MODEL_VERSION} | Dataset: {DATASET_VERSION} | Last Trained: {MODEL_LAST_TRAINED}_")

# Sidebar for File Upload
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file (eswatini_insurance_final_dataset.csv)")

if uploaded_file is None:
    st.info("Upload a file through config", icon="ℹ️")
    st.stop()

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

# Data Preprocessing
missing_values = df.isna().sum().sum()
df['claim_risk'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna('Unknown', inplace=True)

# Convert date columns to numeric features
date_cols = ['policy_start_date', 'claim_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df = df.drop(columns=[col])

# Dynamic Customer Segmentation using K-means
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
X_segment = df[numeric_cols].drop(columns=['claim_risk'], errors='ignore').replace([np.inf, -np.inf], np.nan).dropna()
silhouette = []
range_n_clusters = range(2, 10)
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_segment)
    silhouette.append(silhouette_score(X_segment, labels))
optimal_clusters = range_n_clusters[np.argmax(silhouette)]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['customer_segment'] = kmeans.fit_predict(X_segment).astype(str)

# Define categorical columns
categorical_cols = ['claim_type', 'gender', 'location', 'policy_type', 'insurance_provider', 'customer_segment']
for col in df.columns:
    if df[col].dtype == 'object' and col not in date_cols and col not in ['claim_amount_SZL', 'claim_risk']:
        if col not in categorical_cols:
            categorical_cols.append(col)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Split features and target with balancing
X = df_encoded.drop(columns=['claim_amount_SZL', 'claim_risk'])
y = df_encoded['claim_risk']

# Apply ADASYN oversampling
# Ensure no NaNs or infinite values exist
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()
y = y.loc[X.index]  # Align y with cleaned X

# Apply ADASYN oversampling
adasyn = ADASYN(random_state=42)
X_balanced, y_balanced = adasyn.fit_resample(X, y)

logger.info(f"Shape before balancing: X={X.shape}, y={y.shape}")
logger.info(f"Nulls in X: {X.isnull().sum().sum()}, Nulls in y: {y.isnull().sum()}")


# Split the balanced data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

# Train Random Forest Model with Feature Selection
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight={0: 1.0, 1: 2.5},
    max_depth=15,
    min_samples_leaf=5,
    random_state=42
)
selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
logger.info(f"Selected features: {selected_features.tolist()}")
rf.fit(X_train_selected, y_train)
y_pred_rf = rf.predict(X_test_selected)
report = classification_report(y_test, y_pred_rf, output_dict=True)
recall_class_1 = report['1']['recall']
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test_selected)[:, 1])
roc_auc = auc(fpr, tpr)
if recall_class_1 > 0.39:
    logger.info(f"Model saved. Recall for class 1: {recall_class_1}")
else:
    logger.info(f"Model not saved. Recall for class 1: {recall_class_1} (below 0.39 threshold)")

# Metrics Display with Pie Chart
st.subheader("Key Metrics")
cols = st.columns(5)
metrics = [
    {"label": "Total Records", "value": len(df)},
    {"label": "Model AUC", "value": f"{roc_auc:.2f}"},
    {"label": "Missing Values", "value": df.isnull().sum().sum()},
    {"label": "Selected Features", "value": len(selected_features)}
]
for col, metric in zip(cols[:4], metrics):
    col.metric(label=metric["label"], value=metric["value"])

# Pie Chart of Risk Category Distribution
with cols[4]:
    st.subheader("Risk Category Distribution")
    risk_counts = pd.Series(y_balanced).value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=['Low Risk', 'High Risk'],
        title="Risk Category Distribution",
        color_discrete_sequence=['#00CC96', '#EF553B']
    )
    fig_pie.update_layout(height=200, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# Classification Report
st.subheader("Classification Report")
st.json(report)

st.success("Model trained and results displayed successfully!")

# Map Functions with Segmentations
def init_map():
    eswatini_center = [-26.5225, 31.4659]
    m = folium.Map(location=eswatini_center, zoom_start=7, min_zoom=6, max_zoom=8, tiles="cartodbpositron")
    eswatini_bounds = [
        [-27.3, 30.7],
        [-25.7, 32.2]
    ]
    m.fit_bounds(eswatini_bounds)
    return m

def plot_from_df(df, folium_map, selected_risk_levels, selected_regions, selected_segments):
    region_coords = {
        'Lubombo': (-26.3, 31.8), 'Hhohho': (-26.0, 31.1),
        'Manzini': (-26.5, 31.4), 'Shiselweni': (-27.0, 31.3)
    }
    risk_by_region = df.groupby('location')['claim_risk'].mean().reset_index()
    risk_by_region = risk_by_region[risk_by_region['location'].isin(region_coords.keys())]
    risk_by_region['Latitude'] = risk_by_region['location'].map(lambda x: region_coords[x][0])
    risk_by_region['Longitude'] = risk_by_region['location'].map(lambda x: region_coords[x][1])
try:
    geojson_data = geopandas.read_file("eswatini_regions.geojson")
    folium.Choropleth(
        geo_data=geojson_data,
        name='choropleth',
        data=risk_by_region,
        columns=['location', 'claim_risk'],
        key_on='feature.properties.region',  # Updated to match GeoJSON property
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Claim Risk'
    ).add_to(folium_map)
except FileNotFoundError:
        logger.warning("eswatini_regions.geojson not found. Skipping choropleth layer.")
        st.warning("GeoJSON file for Eswatini regions not found. Map will render without choropleth layer.")

    risk_by_region_segment = df.groupby(['location', 'customer_segment'])['claim_risk'].mean().reset_index()
    risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(region_coords.keys())]
    risk_by_region_segment['Latitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][0])
    risk_by_region_segment['Longitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][1])
    
    if risk_by_region_segment['claim_risk'].nunique() > 1:
        quantiles = risk_by_region_segment['claim_risk'].quantile([0, 0.33, 0.66, 1]).values
        risk_by_region_segment['risk_level'] = pd.cut(risk_by_region_segment['claim_risk'], bins=quantiles, labels=['Low', 'Medium', 'High'], include_lowest=True)
    else:
        risk_by_region_segment['risk_level'] = 'Low'
        logger.info("Not enough variance in claim_risk for binning. Defaulting to 'Low' risk level.")

    if selected_risk_levels:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['risk_level'].isin(selected_risk_levels)]
    if selected_regions:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(selected_regions)]
    if selected_segments:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['customer_segment'].isin(selected_segments)]

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

    df['Latitude'] = df['location'].map(lambda x: region_coords.get(x, (-26.5, 31.5))[0])
    df['Longitude'] = df['location'].map(lambda x: region_coords.get(x, (-26.5, 31.5))[1])
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows() if row['claim_risk'] == 1]
    HeatMap(heat_data, radius=15).add_to(folium_map)
    folium.plugins.MiniMap().add_to(folium_map)
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
        expected_features = selected_features
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

        # Drift Detection with Line Chart
        drift_feature = 'probability_high_risk'
        if os.path.exists(log_file):
            pred_log_df = pd.read_csv(log_file)
            pred_log_df['timestamp'] = pd.to_datetime(pred_log_df['timestamp'])
            recent_preds = pred_log_df[pred_log_df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]
            if len(recent_preds) > 10:
                original_dist = pred_log_df[drift_feature].dropna().values
                recent_dist = recent_preds[drift_feature].dropna().values
                stat, p_value = ks_2samp(original_dist, recent_dist)
                if p_value < 0.05:
                    st.warning(f"Data drift detected in {drift_feature} (p-value: {p_value:.4f}). Consider retraining the model.")
                    logger.warning(f"Data drift detected in {drift_feature} (p-value: {p_value:.4f})")
                else:
                    logger.info(f"No significant drift detected (p-value: {p_value:.4f})")
                
                st.subheader("Risk Drift Over Time")
                fig_drift = px.line(
                    pred_log_df,
                    x='timestamp',
                    y='probability_high_risk',
                    title="Risk Drift Over Time",
                    markers=True,
                    color_discrete_sequence=['#00CC96']
                )
                fig_drift.update_layout(height=300)
                st.plotly_chart(fig_drift, use_container_width=True)
            else:
                st.info("Not enough recent predictions for drift detection.")
        else:
            st.info("No prediction history available for drift detection.")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.error(f"Prediction failed: {str(e)}")

# Section 2: Model Performance and Risk Trends
with st.expander("Model Performance", expanded=True):
    col4, col5 = st.columns(2)
    with col4:
        st.header("Confusion Matrix")
        try:
            y_pred = rf.predict(X_test_selected)
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
            fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test_selected)[:, 1])
            fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'FPR', 'y': 'TPR'})
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'))
            fig_roc.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_roc)
            st.text(f"Recall for High Risk: {recall_class_1:.2f}")
        except Exception as e:
            st.error(f"ROC plotting failed: {str(e)}")

    col6, _ = st.columns([1, 1])
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
                
                st.subheader("Risk Distribution Over Time (Heatmap)")
                monthly_risk = pred_log.groupby(pred_log['timestamp'].dt.strftime('%Y-%m'))['probability_high_risk'].mean().reset_index()
                monthly_risk['risk_score'] = monthly_risk['probability_high_risk']
                fig_heatmap = px.scatter(
                    monthly_risk,
                    x='timestamp',
                    y='risk_score',
                    size='risk_score',
                    color='risk_score',
                    title="Risk Density Over Time",
                    color_continuous_scale='Reds',
                    labels={'timestamp': 'Month', 'risk_score': 'Risk Score'}
                )
                fig_heatmap.update_layout(height=300)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                logger.info("Risk trend and heatmap rendered")
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
            sample_data = X_test.loc[:, selected_features].sample(50, random_state=42)
            shap_values = explainer.shap_values(sample_data.values)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.array(shap_values).reshape(-1, len(selected_features))
            st.subheader("Features Used in SHAP Analysis")
            st.write(list(selected_features))
            fig_shap = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_data.values, feature_names=selected_features, max_display=5, show=False, plot_type="bar")
            if plt.gcf().axes:
                plt.title('Top Features for High Risk')
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.savefig('shap_plot.png')
            shap_df = pd.DataFrame({'Feature': selected_features, 'SHAP Value': np.abs(shap_values).mean(axis=0)}).sort_values(by='SHAP Value', ascending=False).head(5)
            st.session_state['shap_df'] = shap_df
            logger.info("SHAP plot rendered")
        except Exception as e:
            st.error(f"SHAP plot failed: {str(e)}")
            logger.error(f"SHAP plot failed: {str(e)}")

with col8:
    st.header("Risk by Location")
    risk_by_location = df.groupby('location')['claim_risk'].mean().reset_index()
    fig_loc = px.bar(risk_by_location, x='location', y='claim_risk', title="Average Risk by Location (%)",
                     color='claim_risk', color_continuous_scale='Blues', height=300)
    fig_loc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_loc)

with col9:
    st.header("Risk by Claim Type")
    risk_by_claim_type = df.groupby('claim_type')['claim_risk'].mean().reset_index()
    fig_claim = px.bar(risk_by_claim_type, x='claim_type', y='claim_risk', title="Average Risk by Claim Type (%)",
                       color='claim_risk', color_continuous_scale='Blues', height=300)
    fig_claim.update_layout(template="plotly_dark")
    st.plotly_chart(fig_claim)

# Section 4: Segmentation Drill-down with Optimized Visualizations
col10, col11 = st.columns([1, 1])
with col10:
    st.header("Customer Segment Drill-down")
    segment = st.selectbox("Select Customer Segment", df['customer_segment'].unique())
    segment_data = df[df['customer_segment'] == segment]
    try:
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
    
    # Stacked Bar Chart of Risk Factors by Segment (Optimized)
    st.subheader("Risk Factor Contribution by Segment")
    try:
        segment_shap_values = {}
        # Limit to a sample to prevent infinite computation
        sample_size = min(100, len(segment_data))  # Cap at 100 samples
        for seg in df['customer_segment'].unique():
            seg_data_seg = df[df['customer_segment'] == seg].sample(n=sample_size, random_state=42)
            columns_to_drop = ['claim_amount_SZL', 'claim_risk']
            columns_to_drop = [col for col in columns_to_drop if col in seg_data_seg.columns]
            seg_encoded = pd.get_dummies(seg_data_seg.drop(columns=columns_to_drop), columns=categorical_cols, drop_first=False)
            for col in selected_features:
                if col not in seg_encoded.columns:
                    seg_encoded[col] = 0
            seg_encoded = seg_encoded[selected_features]
            if len(seg_encoded) > 0:
                seg_shap = explainer.shap_values(seg_encoded.head(50).values)  # Limit SHAP to 50 rows
                if isinstance(seg_shap, list):
                    seg_shap = seg_shap[1]
                segment_shap_values[seg] = np.abs(seg_shap).mean(axis=0) if seg_shap.size > 0 else np.zeros(len(selected_features))
            else:
                segment_shap_values[seg] = np.zeros(len(selected_features))
        
        shap_by_segment = pd.DataFrame(segment_shap_values, index=selected_features).T
        if not shap_by_segment.empty:
            top_features = shap_by_segment.mean().sort_values(ascending=False).head(2).index
            fig_stacked = px.bar(
                shap_by_segment.reset_index(),
                x='customer_segment',
                y=[top_features[0], top_features[1]],
                title="Top Risk Factors by Segment",
                labels={'value': 'SHAP Contribution', 'customer_segment': 'Segment'},
                barmode='stack',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            fig_stacked.update_layout(height=300)
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.warning("No data available for stacked bar chart.")
    except Exception as e:
        st.warning(f"Stacked bar chart failed: {str(e)}")

    # Radar Chart for Risk Profile Comparison (Optimized)
    st.subheader("Risk Profile Comparison Across Segments")
    try:
        segment_profiles = {}
        sample_size = min(100, len(df))  # Cap at 100 samples
        sampled_df = df.sample(n=sample_size, random_state=42)
        for seg in sampled_df['customer_segment'].unique():
            seg_data = sampled_df[sampled_df['customer_segment'] == seg]
            profile = {
                'Risk Level': seg_data['claim_risk'].mean(),
                'Premium': seg_data['policy_annual_premium'].mean() if 'policy_annual_premium' in seg_data.columns else 0,
                'Age': seg_data['insured_age'].mean() if 'insured_age' in seg_data.columns else 0,
                'Incident Severity': seg_data['incident_severity'].mean() if 'incident_severity' in seg_data.columns else 0
            }
            for key in profile:
                if key == 'Risk Level':
                    profile[key] = profile[key]
                elif key in df.columns:
                    profile[key] = (profile[key] - df[key].min()) / (df[key].max() - df[key].min()) if df[key].max() != df[key].min() else 0
                else:
                    profile[key] = 0
            segment_profiles[seg] = profile
        
        radar_data = []
        for seg, profile in segment_profiles.items():
            radar_data.append({
                'Segment': f"Segment {seg}",
                'Risk Level': profile['Risk Level'],
                'Premium': profile['Premium'],
                'Age': profile['Age'],
                'Incident Severity': profile['Incident Severity']
            })
        radar_df = pd.DataFrame(radar_data)
        if not radar_df.empty:
            fig_radar = px.line_polar(
                radar_df.melt(id_vars='Segment', var_name='Metric', value_name='Value'),
                r='Value',
                theta='Metric',
                color='Segment',
                line_close=True,
                title="Risk Profile Comparison Across Segments",
                color_discrete_sequence=['#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
            )
            fig_radar.update_layout(height=400)
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("No data available for radar chart.")
    except Exception as e:
        st.warning(f"Radar chart failed: {str(e)}")
    logger.info(f"Segment trend and new visualizations rendered for {segment}")

with col11:
    st.header("Top Features for Segment")
    try:
        columns_to_drop = ['claim_amount_SZL', 'claim_risk', 'Latitude', 'Longitude']
        columns_to_drop = [col for col in columns_to_drop if col in segment_data.columns]
        segment_data = segment_data.drop(columns=columns_to_drop)
        segment_encoded = pd.get_dummies(segment_data, columns=categorical_cols, drop_first=False)
        for col in selected_features:
            if col not in segment_encoded.columns:
                segment_encoded[col] = 0
        segment_encoded = segment_encoded[selected_features].values
        shap_values_segment = explainer.shap_values(segment_encoded[:50])  # Limit to 50 rows
        if isinstance(shap_values_segment, list):
            shap_values_segment = shap_values_segment[1]
        shap_values_segment = np.array(shap_values_segment).reshape(-1, len(selected_features))
        fig_shap_segment = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_segment, segment_encoded[:50], feature_names=selected_features, max_display=5, show=False, plot_type="bar")
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
        map_data = folium_static(m, height=500, width=1000)
        selected_region = None
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
                sample_region = sample_region.drop(columns=['claim_amount_SZL', 'claim_risk', 'Latitude', 'Longitude'])
                sample_encoded_region = pd.get_dummies(sample_region, columns=categorical_cols, drop_first=False)
                for col in selected_features:
                    if col not in sample_encoded_region.columns:
                        sample_encoded_region[col] = 0
                sample_encoded_region = sample_encoded_region[selected_features].values
                shap_values_region = explainer.shap_values(sample_encoded_region)
                if isinstance(shap_values_region, list):
                    shap_values_region = shap_values_region[1]
                shap_values_region = np.array(shap_values_region).reshape(-1, len(selected_features))
                fig_shap_region = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_region, sample_encoded_region, feature_names=selected_features, max_display=5, show=False, plot_type="bar")
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

if st.button("Download PDF Report"):
    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name="report.pdf")

# Notes
st.markdown("**Note**: Ensure the dataset is available. Risk map uses claim risk data to highlight high-risk areas.", unsafe_allow_html=True)
st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
