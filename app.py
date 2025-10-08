# --- Import Libraries --- 
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import altair as alt
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SHAPe Avocado Dashboard",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stSlider [data-baseweb="slider"] {
            padding: 0;
        }
        .metric-card {
            border-radius: 10px;
            padding: 15px;
            background-color: #f0f2f6;
            margin-bottom: 15px;
        }
        .analysis-card {
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #2ecc71;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Yield Reference Data ---
YIELD_REFERENCE = {
    '0-3': {'fruits': 275, 'kg': 45.8},
    '4-7': {'fruits': 350, 'kg': 58.3},
    '8+': {'fruits': 900, 'kg': 150.0}
}

# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_farmer_data():
    """Load the farmer baseline data"""
    try:
        df = pd.read_excel('shape_data.xlsx', sheet_name='Baseline')
        
        # Clean and preprocess data
        if 'data_time' in df.columns:
            df['submitdate'] = pd.to_datetime(df['data_time'], errors='coerce')
        
        # Handle area under cultivation
        area_col = '2.2 Total Area under Avocado Cultivation (Acres)'
        if area_col in df.columns:
            df['Total Area under Avocado Cultivation (Acres)'] = pd.to_numeric(df[area_col], errors='coerce')
        
        # Handle tree count
        trees_col = '2.3 Number of Avocado Trees Planted'
        if trees_col in df.columns:
            df['Number of Avocado Trees Planted'] = pd.to_numeric(df[trees_col], errors='coerce')
        
        # Calculate yields for different age groups if columns exist
        age_groups = {
            '0-3': '4.8 Average No. of Fruits per avocado tree aged 0-3 years',
            '4-7': '4.81 Average No. of Fruits per avocado tree aged 4-7 years',
            '8+': '4.82 Average No. of Fruits per avocado tree aged 8+ years'
        }
        
        for age, col in age_groups.items():
            if col in df.columns:
                df[f'Fruits per tree {age} years'] = pd.to_numeric(df[col], errors='coerce')
                # Calculate kg based on reference data
                df[f'Yield (kg/tree) {age} years'] = df[f'Fruits per tree {age} years'] * (YIELD_REFERENCE[age]['kg'] / YIELD_REFERENCE[age]['fruits'])
        
        # Handle price outliers for Hass variety
        if '5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)' in df.columns:
            df['5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'] = pd.to_numeric(
                df['5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'], errors='coerce')
            df['5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'] = df['5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'].apply(
                lambda x: x if x <= 120 else np.nan)
        
        # Merge loss reasons
        if '4.31  Primary Cause of Loss last season' in df.columns and '4.32 Other Causes of Loss last season' in df.columns:
            df['Combined Loss Reasons'] = df['4.31  Primary Cause of Loss last season'].fillna('') + '; ' + df['4.32 Other Causes of Loss last season'].fillna('')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading farmer data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_exporter_metrics():
    """Load the exporter metrics data"""
    try:
        df = pd.read_excel('shape_data.xlsx', sheet_name='Metrics')
        
        # Check if required columns exist
        if 'Metrics' not in df.columns:
            st.warning("Metrics sheet doesn't have the expected structure")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.warning(f"Couldn't load exporter metrics: {str(e)}")
        return pd.DataFrame()

# --- Advanced Analysis Functions ---
class ShapeAdvancedAnalytics:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def prepare_certification_data(self):
        """Prepare data for certification prediction"""
        if self.df.empty:
            return None, None
            
        # Features for certification prediction
        cert_features = [
            '2.1 Total Farm Size (Acres)', '2.3 Number of Avocado Trees Planted',
            '1.14 Experience in Avocado farming in years', '3.2 Are Good Agricultural Practices (GAP) applied in the orchard?',
            '3.8  Is an Integrated Pest Management (IPM) program implemented?', '6.4 Use of Clean Harvesting Tools',
            '6.5 Proper Disposal of Waste', '6.6 Compliance with Pesticide Withdrawal Period (REI/ PHI)',
            '6.7 Use of Approved Pesticides Only', '6.8 Application of Correct Dosage'
        ]
        
        # Target - GACC approval
        if '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status' not in self.df.columns:
            return None, None
            
        X = self.df[cert_features].copy()
        y = self.df['1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status']
        
        # Convert yes/no to binary
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}).fillna(0)
        
        y = y.map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}).fillna(0)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def predict_certification_success(self):
        """Predict certification success probability"""
        X, y = self.prepare_certification_data()
        if X is None or len(X) < 10:
            return None
            
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        return {
            'model': model,
            'features': X.columns.tolist(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
    
    def analyze_yield_drivers(self):
        """Analyze key drivers of yield"""
        if self.df.empty or '4.2 Total Harvest Last Season (kg)' not in self.df.columns:
            return None
            
        yield_features = [
            '2.1 Total Farm Size (Acres)', '2.3 Number of Avocado Trees Planted',
            '1.14 Experience in Avocado farming in years', '3.3 Type of Fertilizer Used/Organic',
            '3.3 Type of Fertilizer Used/Inorganic', '3.5 Irrigation Practices/Rainfed',
            '3.5 Irrigation Practices/Drip', '3.5 Irrigation Practices/Sprinkler',
            '3.4 Soil Conservation Measures Applied/Mulching', '3.4 Soil Conservation Measures Applied/Terracing'
        ]
        
        X = self.df[yield_features].copy()
        y = self.df['4.2 Total Harvest Last Season (kg)']
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        if len(X) < 10:
            return None
            
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'features': X.columns.tolist(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            'r2_score': r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def segment_farmers(self):
        """Segment farmers into clusters"""
        if self.df.empty:
            return None
            
        segmentation_features = [
            '2.1 Total Farm Size (Acres)', '2.3 Number of Avocado Trees Planted',
            '1.14 Experience in Avocado farming in years', '4.2 Total Harvest Last Season (kg)',
            '5.3 Total Income from Avocado Sales (KSH last season)', '3.2 Are Good Agricultural Practices (GAP) applied in the orchard?'
        ]
        
        X = self.df[segmentation_features].copy()
        
        # Convert categorical to numerical
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}).fillna(0)
        
        X = X.fillna(X.mean())
        
        if len(X) < 10:
            return None
            
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        return {
            'clusters': clusters,
            'centers': kmeans.cluster_centers_,
            'features': X.columns.tolist(),
            'cluster_labels': ['Small Traditional', 'Emerging Commercial', 'Established Commercial', 'High-Performance']
        }
    
    def economic_analysis(self):
        """Perform economic analysis"""
        if self.df.empty or '5.3 Total Income from Avocado Sales (KSH last season)' not in self.df.columns:
            return None
        
        # Calculate ROI metrics with better error handling
        economic_data = self.df[['2.1 Total Farm Size (Acres)', '5.3 Total Income from Avocado Sales (KSH last season)']].copy()
    
        # Convert to numeric and handle non-numeric values
        economic_data['2.1 Total Farm Size (Acres)'] = pd.to_numeric(economic_data['2.1 Total Farm Size (Acres)'], errors='coerce')
        economic_data['5.3 Total Income from Avocado Sales (KSH last season)'] = pd.to_numeric(
            economic_data['5.3 Total Income from Avocado Sales (KSH last season)'], errors='coerce')
    
        # Remove rows with missing values or zero farm size
        economic_data = economic_data.dropna()
        economic_data = economic_data[economic_data['2.1 Total Farm Size (Acres)'] > 0]
    
        if len(economic_data) < 5:
            return None
        
        # Calculate income per acre with division by zero protection
        economic_data['Income_per_Acre'] = (
            economic_data['5.3 Total Income from Avocado Sales (KSH last season)'] / 
            economic_data['2.1 Total Farm Size (Acres)']
        )
    
        # Remove extreme outliers (top and bottom 1%)
        q_low = economic_data['Income_per_Acre'].quantile(0.01)
        q_hi = economic_data['Income_per_Acre'].quantile(0.99)
        economic_data = economic_data[(economic_data['Income_per_Acre'] >= q_low) & 
                                    (economic_data['Income_per_Acre'] <= q_hi)]
    
        if len(economic_data) < 5:
            return None
    
        return {
            'income_per_acre': economic_data['Income_per_Acre'],
            'stats': {
                'mean': economic_data['Income_per_Acre'].mean(),
                'median': economic_data['Income_per_Acre'].median(),
                'std': economic_data['Income_per_Acre'].std(),
                'q1': economic_data['Income_per_Acre'].quantile(0.25),
                'q3': economic_data['Income_per_Acre'].quantile(0.75)
            }
        }
    
    def correlation_analysis(self):
        """Perform comprehensive correlation analysis"""
        if self.df.empty:
            return None
            
        # Select key numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        important_cols = [col for col in numeric_cols if any(x in col for x in 
                          ['Yield', 'Income', 'Trees', 'Size', 'Experience', 'Price'])]
        
        if len(important_cols) < 3:
            important_cols = numeric_cols[:10]  # Fallback to first 10 numeric columns
            
        corr_data = self.df[important_cols].corr()
        
        return corr_data

# --- Visualization Functions ---
def create_farm_map(df):
    """Create interactive map of farms"""
    if df.empty or '_1.21 GPS Coordinates of Orchard_latitude' not in df.columns:
        return None
    
    # Filter valid coordinates
    map_df = df.dropna(subset=['_1.21 GPS Coordinates of Orchard_latitude', '_1.21 GPS Coordinates of Orchard_longitude'])
    
    if map_df.empty:
        return None
    
    # Create base map centered on Kenya
    m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)
    
    # Add markers for each farm
    for idx, row in map_df.iterrows():
        farm_name = row.get("1.22 Orchard Name/ Name of farm", "N/A")
        farmer_name = row.get("1.10 Farmer's Name (Three Names)", "N/A")
        trees = row.get("2.3 Number of Avocado Trees Planted", "N/A")
        variety = "Hass" if row.get("3.1 Variety Grown/Hass", 0) == 1 else "Other"

        popup_text = f"""
        <b>Farm:</b> {farm_name}<br>
        <b>Farmer:</b> {farmer_name}<br>
        <b>Trees:</b> {trees}<br>
        <b>Variety:</b> {variety}
        """

        folium.Marker(
            location=[row["_1.21 GPS Coordinates of Orchard_latitude"],
                    row["_1.21 GPS Coordinates of Orchard_longitude"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)
    
    return m

def create_certification_chart(df):
    """Create certification status chart"""
    cert_cols = {
        'GlobalGAP': '6.2 Which Certifications is the Orchard compliant for this season?/Global GAP',
        'Organic': '6.2 Which Certifications is the Orchard compliant for this season?/Organic',
        'FairTrade': '6.2 Which Certifications is the Orchard compliant for this season?/FairTrade',
        'China': '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status'
    }
    
    cert_data = []
    for cert, col in cert_cols.items():
        if col in df.columns:
            if cert == 'China':
                count = (df[col] == 'Yes').sum()
            else:
                count = df[col].sum() if df[col].dtype in [int, float] else (df[col] == 1).sum()
            cert_data.append({'Certification': cert, 'Count': count})
    
    if not cert_data:
        return None
    
    cert_df = pd.DataFrame(cert_data)
    
    chart = alt.Chart(cert_df).mark_bar().encode(
        x='Certification:N',
        y='Count:Q',
        color=alt.Color('Certification:N', scale=alt.Scale(scheme='greens')),
        tooltip=['Certification', 'Count']
    ).properties(
        title='Farm Certification Status',
        width=600,
        height=400
    )
    
    return chart

def create_yield_comparison_chart(df):
    """Create yield comparison by tree age with grouped bars + value & % labels."""
    if df.empty:
        return None

    dark_blue  = '#0057D9'  # Actual
    light_blue = '#8EC4FF'  # Expected

    # Build tidy data + keep age order
    age_order = ['0-3 years', '4-7 years', '8+ years']
    rows = []
    for age in ['0-3', '4-7', '8+']:
        col = f'Fruits per tree {age} years'
        if col in df.columns:
            actual = float(pd.to_numeric(df[col], errors='coerce').mean())
            expected = float(YIELD_REFERENCE[age]['fruits'])
            label = f'{age} years'
            rows += [
                {'Age Group': label, 'Category': 'Actual',   'Value': actual,   'Expected': expected},
                {'Age Group': label, 'Category': 'Expected', 'Value': expected, 'Expected': expected},
            ]
    if not rows:
        return None

    data = pd.DataFrame(rows)
    # Compute "% of expected" only for Actual bars
    data['PctOfExpected'] = np.where(
        data['Category'].eq('Actual') & (data['Expected'] > 0),
        (data['Value'] / data['Expected']) * 100.0,
        np.nan
    )
    data['PctLabel'] = data['PctOfExpected'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")

    base = alt.Chart(data)

    bar = base.mark_bar(opacity=0.85, stroke='black', strokeWidth=0.5).encode(
        x=alt.X('Age Group:N', title='Age Group', sort=age_order, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset('Category:N'),
        y=alt.Y('Value:Q',
                title='Average Fruits per Tree',
                axis=alt.Axis(format=',')  # thousands separators
               ),
        color=alt.Color(
            'Category:N',
            title='Measurement Type',
            scale=alt.Scale(domain=['Actual', 'Expected'], range=[dark_blue, light_blue])
        ),
        tooltip=['Age Group', 'Category', alt.Tooltip('Value:Q', format=',.0f')]
    ).properties(width=380, height=380)

    # Value label on every bar
    value_labels = base.mark_text(dy=-6, fontWeight='bold').encode(
        x='Age Group:N',
        xOffset='Category:N',
        y='Value:Q',
        text=alt.Text('Value:Q', format=',.0f'),
        color=alt.value('black')
    )

    # % of expected shown only on Actual bars (slightly above the value label)
    pct_labels = base.transform_filter(
        alt.datum.Category == 'Actual'
    ).mark_text(dy=-22).encode(
        x='Age Group:N',
        xOffset='Category:N',
        y='Value:Q',
        text='PctLabel:N',
        color=alt.value('#555')
    )

    chart = (bar + value_labels + pct_labels).properties(
        title='Average Fruits per Tree (Actual vs Expected)'
    ).configure_view(
        stroke='transparent'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

    return chart

def create_wordcloud(text, title):
    """Generate word cloud from text"""
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    return fig

# --- Advanced Analysis Visualization Functions ---
def plot_feature_importance(importance_dict, title):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 title=title, orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_cluster_profiles(cluster_results, df):
    """Plot cluster profiles"""
    if cluster_results is None:
        return None
        
    # Create cluster profiles dataframe
    profile_data = []
    for i, label in enumerate(cluster_results['cluster_labels']):
        profile_data.append({
            'Cluster': label,
            **{feat: cluster_results['centers'][i][j] for j, feat in enumerate(cluster_results['features'])}
        })
    
    profile_df = pd.DataFrame(profile_data)
    
    # Create radar chart for cluster profiles
    fig = go.Figure()
    
    for i, row in profile_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[1:].values,
            theta=row[1:].index,
            fill='toself',
            name=row['Cluster']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Farmer Cluster Profiles",
        showlegend=True
    )
    
    return fig

def plot_correlation_heatmap(corr_data, title):
    """Plot correlation heatmap"""
    fig = px.imshow(corr_data, 
                   title=title,
                   color_continuous_scale='RdBu_r',
                   aspect="auto")
    return fig

# =========================
# --- Investor View (NEW) ---
# =========================

def _safe_num(s):
    return pd.to_numeric(s, errors='coerce')

def _prep_age_density_fields(df, density_threshold):
    """Prepare fields needed for investor view & forecasting."""
    d = df.copy()

    # Area & total trees
    area_col = 'Total Area under Avocado Cultivation (Acres)'
    trees_col = '2.3 Number of Avocado Trees Planted'
    d[area_col] = _safe_num(d.get(area_col))
    d[trees_col] = _safe_num(d.get(trees_col))

    # Age class tree counts
    d['trees_0_3'] = _safe_num(d.get('2.41 Number of trees for Age class 0-3 years')).fillna(0)
    d['trees_4_7'] = _safe_num(d.get('2.42 Number of trees for Age class 4-7 years')).fillna(0)
    d['trees_8_plus'] = _safe_num(d.get('2.43 Number of trees for Age class 8+ years')).fillna(0)

    # Density (trees per acre) using planted trees / area-under-avocado
    with np.errstate(divide='ignore', invalid='ignore'):
        d['trees_per_acre'] = d[trees_col] / d[area_col]

    # Density bin text uses the chosen threshold
    low_lbl = f"≤{int(density_threshold)}"
    high_lbl = f">{int(density_threshold)}"
    d['density_bin'] = np.where(d['trees_per_acre'] > density_threshold, high_lbl, low_lbl)

    # Dominant age group by count
    age_counts = d[['trees_0_3','trees_4_7','trees_8_plus']]
    dom = age_counts.idxmax(axis=1).fillna('trees_0_3')
    d['dominant_age_group'] = dom.map({
        'trees_0_3':'0-3 years',
        'trees_4_7':'4-7 years',
        'trees_8_plus':'8+ years'
    })

    # Income per acre
    income_col = '5.3 Total Income from Avocado Sales (KSH last season)'
    d[income_col] = _safe_num(d.get(income_col))
    with np.errstate(divide='ignore', invalid='ignore'):
        d['income_per_acre'] = d[income_col] / d[area_col]

    return d

def show_investor_income_view(df):
    """Income per acre segmented by dominant age group and density, plus tree age proportions."""
    st.subheader("💰 Income by Tree Age & Density")

    # Choose threshold (default to 90 if available; else median from data)
    tmp = df.copy()
    area_col = 'Total Area under Avocado Cultivation (Acres)'
    trees_col = '2.3 Number of Avocado Trees Planted'
    tmp[area_col] = _safe_num(tmp.get(area_col))
    tmp[trees_col] = _safe_num(tmp.get(trees_col))
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp['trees_per_acre'] = tmp[trees_col] / tmp[area_col]
    default_threshold = 90.0 if np.isfinite(tmp['trees_per_acre']).any() else 90.0
    density_threshold = st.number_input(
        "Density threshold (trees/acre) for binning",
        min_value=1.0,
        value=float(default_threshold),
        help="Farms at or below this threshold are grouped in the lower density bin; those above in the higher bin."
    )

    d = _prep_age_density_fields(df, density_threshold)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=['income_per_acre', 'dominant_age_group', 'density_bin', 'trees_per_acre'])

    if d.empty:
        st.info("Not enough data to compute investor view.")
        return

    col1, col2 = st.columns(2)

    # A) Average income/acre by dominant age group × density
    grp = (d.groupby(['dominant_age_group','density_bin'])['income_per_acre']
             .agg(['count','mean','median'])
             .reset_index()
             .rename(columns={'mean':'avg_income_per_acre','median':'median_income_per_acre'}))

    with col1:
        st.markdown("**Income per Acre by Dominant Tree Age & Density**")
        chart1 = alt.Chart(grp).mark_bar().encode(
            x=alt.X('dominant_age_group:N', title='Dominant Age Group'),
            y=alt.Y('avg_income_per_acre:Q', title='Average Income per Acre (KSh)'),
            color=alt.Color('density_bin:N', title='Density (trees/acre)'),
            tooltip=['dominant_age_group','density_bin','count',
                     alt.Tooltip('avg_income_per_acre:Q', format=',.0f', title='Avg Income/acre'),
                     alt.Tooltip('median_income_per_acre:Q', format=',.0f', title='Median Income/acre')]
        ).properties(height=380)
        st.altair_chart(chart1, use_container_width=True)

    # B) Proportion of planted trees by age category (portfolio mix)
    total_trees = pd.DataFrame({
        'Age Group':['0-3 years','4-7 years','8+ years'],
        'Trees':[
            d['trees_0_3'].sum(),
            d['trees_4_7'].sum(),
            d['trees_8_plus'].sum()
        ]
    })
    total_trees = total_trees[total_trees['Trees'] > 0]

    with col2:
        st.markdown("**Planted Trees by Age Category (Share)**")
        chart2 = alt.Chart(total_trees).mark_arc().encode(
            theta=alt.Theta('Trees:Q', stack=True),
            color=alt.Color('Age Group:N'),
            tooltip=['Age Group', alt.Tooltip('Trees:Q', format=',')]
        ).properties(height=380)
        st.altair_chart(chart2, use_container_width=True)

    # Small table for reference
    with st.expander("Show summary table"):
        st.dataframe(grp.sort_values(['dominant_age_group','density_bin']))

def show_income_potential_forecast(df):
    """Transparent income potential forecast + sensitivity analysis."""
    st.subheader("📈 Income Potential Forecast (kg/tree × trees/acre × price)")

    # Prices: compute from data; if none, ask user to input.
    price_col = '5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'
    prices = pd.to_numeric(df.get(price_col), errors='coerce') if price_col in df.columns else pd.Series(dtype=float)
    prices = prices[(~prices.isna()) & (prices > 0) & (prices <= prices.quantile(0.99) if len(prices) else True)]
    if len(prices):
        median_price = float(prices.median())
    else:
        median_price = st.number_input(
            "Enter a reference price (KSh/kg):",
            min_value=0.0, value=0.0
        )

    # Density threshold control (align with investor view)
    area_col = 'Total Area under Avocado Cultivation (Acres)'
    trees_col = '2.3 Number of Avocado Trees Planted'
    tmp = df.copy()
    tmp[area_col] = _safe_num(tmp.get(area_col))
    tmp[trees_col] = _safe_num(tmp.get(trees_col))
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp['trees_per_acre'] = tmp[trees_col] / tmp[area_col]
    default_threshold = 90.0 if np.isfinite(tmp['trees_per_acre']).any() else 90.0
    density_threshold = st.number_input(
        "Forecast density threshold (trees/acre)",
        min_value=1.0,
        value=float(default_threshold),
        help="Used to label farms into low/high density buckets in the forecast charts."
    )

    d = _prep_age_density_fields(df, density_threshold).copy()
    d = d.dropna(subset=[area_col])
    d = d[d[area_col] > 0].replace([np.inf, -np.inf], np.nan)

    # Trees per acre by age class
    for col_src, col_out in [
        ('trees_0_3','tpa_0_3'),
        ('trees_4_7','tpa_4_7'),
        ('trees_8_plus','tpa_8_plus'),
    ]:
        d[col_out] = d[col_src] / d[area_col]

    # Reference kg per tree from YIELD_REFERENCE
    kg03 = YIELD_REFERENCE['0-3']['kg']
    kg47 = YIELD_REFERENCE['4-7']['kg']
    kg8p = YIELD_REFERENCE['8+']['kg']

    # Baseline forecast income per acre
    d['forecast_income_per_acre'] = (
        d['tpa_0_3'] * kg03 +
        d['tpa_4_7'] * kg47 +
        d['tpa_8_plus'] * kg8p
    ) * median_price

    d = d.dropna(subset=['forecast_income_per_acre', 'density_bin', 'dominant_age_group'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Forecast income per acre** (reference price ≈ KSh {median_price:,.0f})")
        st.plotly_chart(
            px.histogram(d, x='forecast_income_per_acre', nbins=30,
                         labels={'forecast_income_per_acre':'Forecast Income per Acre (KSh)'},
                         title="Baseline Forecast Distribution"),
            use_container_width=True
        )

    # By density & dominant age group
    by_cut = (d.groupby(['dominant_age_group','density_bin'])['forecast_income_per_acre']
                .agg(['count','mean','median'])
                .reset_index()
                .rename(columns={'mean':'avg_forecast_income_per_acre',
                                 'median':'median_forecast_income_per_acre'}))

    with col2:
        st.markdown("**Baseline Forecast by Dominant Age × Density**")
        st.altair_chart(
            alt.Chart(by_cut).mark_bar().encode(
                x=alt.X('dominant_age_group:N', title='Dominant Age Group'),
                y=alt.Y('avg_forecast_income_per_acre:Q', title='Avg Forecast Income per Acre (KSh)'),
                color=alt.Color('density_bin:N', title='Density (trees/acre)'),
                tooltip=['dominant_age_group','density_bin','count',
                         alt.Tooltip('avg_forecast_income_per_acre:Q', format=',.0f'),
                         alt.Tooltip('median_forecast_income_per_acre:Q', format=',.0f')]
            ).properties(height=380),
            use_container_width=True
        )

    # Headline KPIs (baseline)
    k1, k2, k3 = st.columns(3)
    k1.metric("Median Forecast (KSh/acre)", f"{d['forecast_income_per_acre'].median():,.0f}")
    k2.metric("P75 Forecast (KSh/acre)", f"{d['forecast_income_per_acre'].quantile(0.75):,.0f}")
    k3.metric("Top Decile (KSh/acre)", f"{d['forecast_income_per_acre'].quantile(0.90):,.0f}")

    # --- Sensitivity widget (price ±%, density scenarios) ---
    st.markdown("### 🎛️ Sensitivity Analysis")
    sens_col1, sens_col2, sens_col3 = st.columns(3)

    with sens_col1:
        price_delta_pct = st.slider(
            "Price change (%)",
            min_value=-50, max_value=50, value=0, step=1,
            help="Apply a percentage change to the reference price to see upside/downside."
        )

    # default target density set from the data distribution (75th percentile of trees_per_acre)
    observed_tpa = d['trees_per_acre'].dropna()
    default_target_density = float(np.nanpercentile(observed_tpa, 75)) if len(observed_tpa) else 0.0
    with sens_col2:
        target_density = st.number_input(
            "Target density (trees/acre)",
            min_value=0.0, value=default_target_density,
            help="Scenario density used to simulate what happens if farms reach this trees/acre level."
        )

    with sens_col3:
        apply_scope = st.selectbox(
            "Apply density scenario to:",
            options=["All farms", f"Only {d['density_bin'].unique()[0]} bin", f"Only {d['density_bin'].unique()[-1]} bin"],
            help="Choose where to apply the target density."
        )

    # Apply sensitivity: price change and density scenario
    price_factor = 1.0 + (price_delta_pct / 100.0)
    scenario_price = median_price * price_factor

    scen = d.copy()

    # Determine which rows to update for density scenario
    if apply_scope == "All farms":
        mask = pd.Series(True, index=scen.index)
    else:
        # extract the label inside the option text after "Only "
        wanted_bin = apply_scope.replace("Only ", "")
        mask = scen['density_bin'].astype(str).eq(wanted_bin)

    # Recalculate trees-per-acre by age so that total trees_per_acre equals target_density (keep within-farm age mix)
    total_tpa = scen[['tpa_0_3','tpa_4_7','tpa_8_plus']].sum(axis=1)
    share_03 = np.divide(scen['tpa_0_3'], total_tpa, out=np.zeros_like(scen['tpa_0_3']), where=total_tpa>0)
    share_47 = np.divide(scen['tpa_4_7'], total_tpa, out=np.zeros_like(scen['tpa_4_7']), where=total_tpa>0)
    share_8p = np.divide(scen['tpa_8_plus'], total_tpa, out=np.zeros_like(scen['tpa_8_plus']), where=total_tpa>0)

    scen.loc[mask, 'tpa_0_3_scen'] = share_03[mask] * target_density
    scen.loc[mask, 'tpa_4_7_scen'] = share_47[mask] * target_density
    scen.loc[mask, 'tpa_8_plus_scen'] = share_8p[mask] * target_density

    # For rows not in mask, keep baseline
    scen.loc[~mask, 'tpa_0_3_scen'] = scen.loc[~mask, 'tpa_0_3']
    scen.loc[~mask, 'tpa_4_7_scen'] = scen.loc[~mask, 'tpa_4_7']
    scen.loc[~mask, 'tpa_8_plus_scen'] = scen.loc[~mask, 'tpa_8_plus']

    # Scenario income/acre
    scen['forecast_income_per_acre_scen'] = (
        scen['tpa_0_3_scen'] * kg03 +
        scen['tpa_4_7_scen'] * kg47 +
        scen['tpa_8_plus_scen'] * kg8p
    ) * scenario_price

    # KPI deltas
    base_med = d['forecast_income_per_acre'].median()
    scen_med = scen['forecast_income_per_acre_scen'].median()
    base_p75 = d['forecast_income_per_acre'].quantile(0.75)
    scen_p75 = scen['forecast_income_per_acre_scen'].quantile(0.75)

    dk1, dk2 = st.columns(2)
    dk1.metric("Median Forecast (Scenario)", f"{scen_med:,.0f}", f"{(scen_med-base_med):+.0f}")
    dk2.metric("P75 Forecast (Scenario)", f"{scen_p75:,.0f}", f"{(scen_p75-base_p75):+.0f}")

    # ---- FIXED: reshape DataFrame BEFORE plotting (no .melt() on a figure) ----
    compare_df = pd.DataFrame({
        'Baseline': d['forecast_income_per_acre'].values,
        'Scenario': scen['forecast_income_per_acre_scen'].values
    })
    compare_long = compare_df.melt(var_name='Case', value_name='KSh_per_acre')
    fig_compare = px.histogram(
        compare_long,
        x='KSh_per_acre',
        color='Case',
        barmode='overlay',
        nbins=30,
        labels={'KSh_per_acre':'KSh/acre'},
        title='Baseline vs Scenario: Income per Acre'
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# --- Dashboard Sections ---
def show_overview(df, metrics_df):
    """Show overview/KPI cards"""
    st.subheader("Program Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Total farmers
    total_farmers = len(df)
    col1.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Farmers</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_farmers}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Total area under cultivation
    total_area = df['Total Area under Avocado Cultivation (Acres)'].sum()
    col2.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Area (Acres)</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_area:,.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Total trees
    total_trees = df['Number of Avocado Trees Planted'].sum()
    col3.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">Total Trees</h3>
        <p style="margin:0; padding:0; font-size:24px">{total_trees:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GACC approved farms
    gacc_col = '1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status'
    gacc_approved = df[gacc_col].eq('Yes').sum() if gacc_col in df.columns else 0
    col4.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; padding:0">China-Approved Farms</h3>
        <p style="margin:0; padding:0; font-size:24px">{gacc_approved}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics comparison - only show if we have metrics data
    if not metrics_df.empty and 'Metrics' in metrics_df.columns:
        st.subheader("Progress vs Targets")
        
        # Get available metrics
        available_metrics = [m for m in ['# of Farms certified', '# of farmers', 
                                       'Land size under Hass avocado in Acres'] 
                           if m in metrics_df['Metrics'].values]
        
        if available_metrics:
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                options=metrics_df['Metrics'].unique(),
                default=available_metrics[:3]  # Show first 3 by default
            )
            
            if selected_metrics:
                filtered_metrics = metrics_df[metrics_df['Metrics'].isin(selected_metrics)]
                
                # Check which periods we have data for
                periods = [p for p in ['Feb/Baseline', 'Total', 'Target'] 
                         if p in metrics_df.columns]
                
                if periods:
                    # Melt dataframe for Altair
                    melted_df = filtered_metrics.melt(id_vars='Metrics', 
                                                    value_vars=periods,
                                                    var_name='Period', 
                                                    value_name='Value')
                    
                    # Create chart
                    chart = alt.Chart(melted_df).mark_bar().encode(
                        x='Metrics:N',
                        y='Value:Q',
                        color='Period:N',
                        column='Period:N',
                        tooltip=['Metrics', 'Period', 'Value']
                    ).properties(
                        width=200,
                        height=400
                    )
                    
                    st.altair_chart(chart)
    else:
        st.info("Exporter metrics data not available or doesn't match expected format")

def show_geospatial(df):
    """Show farm location map"""
    st.subheader("Farm Locations")
    
    if not df.empty:
        m = create_farm_map(df)
        if m:
            folium_static(m, width=1000, height=600)
        else:
            st.warning("No valid geographic coordinates found in the data")
    else:
        st.warning("No data available for mapping")

def show_certification(df):
    """Show certification status"""
    st.subheader("Certification Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cert_chart = create_certification_chart(df)
        if cert_chart:
            st.altair_chart(cert_chart, use_container_width=True)
        else:
            st.warning("No certification data available")
    
    with col2:
        # Show certification requirements checklist
        st.markdown("**China Market Requirements Checklist**")
        
        requirements = [
            ("Farm registration with KEPHIS", "1.25 KEPHIS Registration Status"),
            ("GACC approval", "1.26 General Administration of Customs of the Peoples Republic of China (GACC ) Approval Status"),
            ("Pest monitoring records", "3.81 Pest Monitoring"),
            ("Sanitation records", "3.6 Is there a record of sanitation conditions?"),
            ("Approved pesticide use", "6.7 Use of Approved Pesticides Only")
        ]
        
        for req, col in requirements:
            if col in df.columns:
                compliant = df[col].eq('Yes').sum()
                total = len(df)
                st.progress(compliant/total, text=f"{req}: {compliant}/{total} farms")
            else:
                st.text(f"{req}: Data not available")

def show_production_metrics(df):
    """Show production metrics"""
    st.subheader("Production Metrics")
    
    tab1, tab2, tab3 = st.tabs(["Yields", "Inputs", "Losses"])
    
    with tab1:
        yield_chart = create_yield_comparison_chart(df)
        if yield_chart:
            st.altair_chart(yield_chart, use_container_width=True)
        else:
            st.warning("No yield data available")
        
        # Add variety distribution
        if '3.1 Variety Grown/Hass' in df.columns:
            varieties = ['Hass', 'Fuerte', 'Pinkerton', 'Other']
            variety_counts = {v: df[f'3.1 Variety Grown/{v}'].sum() for v in varieties if f'3.1 Variety Grown/{v}' in df.columns}
            
            if variety_counts:
                variety_df = pd.DataFrame.from_dict(variety_counts, orient='index', columns=['Count']).reset_index()
                variety_df.columns = ['Variety', 'Count']
                
                variety_chart = alt.Chart(variety_df).mark_arc().encode(
                    theta='Count:Q',
                    color='Variety:N',
                    tooltip=['Variety', 'Count']
                ).properties(
                    title='Avocado Varieties Grown',
                    width=400,
                    height=400
                )
                
                st.altair_chart(variety_chart)
    
    with tab2:
        # Input usage visualization
        if '3.3 Type of Fertilizer Used/Organic' in df.columns:
            fertilizer_data = {
                'Type': ['Organic', 'Inorganic', 'None'],
                'Count': [
                    df['3.3 Type of Fertilizer Used/Organic'].sum(),
                    df['3.3 Type of Fertilizer Used/Inorganic'].sum(),
                    df['3.3 Type of Fertilizer Used/None'].sum()
                ]
            }
            
            fertilizer_df = pd.DataFrame(fertilizer_data)
            
            fertilizer_chart = alt.Chart(fertilizer_df).mark_bar().encode(
                x='Type:N',
                y='Count:Q',
                color='Type:N',
                tooltip=['Type', 'Count']
            ).properties(
                title='Fertilizer Usage',
                width=600,
                height=400
            )
            
            st.altair_chart(fertilizer_chart)
        
        # Irrigation practices
        if '3.5 Irrigation Practices/Rainfed' in df.columns:
            irrigation_data = {
                'Method': ['Rainfed', 'Manual', 'Drip', 'Sprinkler'],
                'Count': [
                    df['3.5 Irrigation Practices/Rainfed'].sum(),
                    df['3.5 Irrigation Practices/Manual Watering'].sum(),
                    df['3.5 Irrigation Practices/Drip'].sum() if '3.5 Irrigation Practices/Drip' in df.columns else 0,
                    df['3.5 Irrigation Practices/Sprinkler'].sum() if '3.5 Irrigation Practices/Sprinkler' in df.columns else 0
                ]
            }
            
            irrigation_df = pd.DataFrame(irrigation_data)
            
            irrigation_chart = alt.Chart(irrigation_df).mark_bar().encode(
                x='Method:N',
                y='Count:Q',
                color='Method:N',
                tooltip=['Method', 'Count']
            ).properties(
                title='Irrigation Methods',
                width=600,
                height=400
            )
            
            st.altair_chart(irrigation_chart)
    
    with tab3:
        # Post-harvest losses
        if 'Combined Loss Reasons' in df.columns:
            loss_reasons = df['Combined Loss Reasons'].str.split(';').explode().str.strip()
            loss_reasons = loss_reasons[loss_reasons != ''].value_counts().reset_index()
            loss_reasons.columns = ['Reason', 'Count']
            
            if not loss_reasons.empty:
                loss_chart = alt.Chart(loss_reasons).mark_bar().encode(
                    x='Count:Q',
                    y='Reason:N',
                    color='Reason:N',
                    tooltip=['Reason', 'Count']
                ).properties(
                    title='Primary Causes of Post-Harvest Loss',
                    width=600,
                    height=400
                )
                
                st.altair_chart(loss_chart)

def show_market_analysis(df):
    """Show market and income analysis"""
    st.subheader("Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market outlets
        if '5.1 Main Market Outlet' in df.columns:
            market_counts = df['5.1 Main Market Outlet'].value_counts().reset_index()
            market_counts.columns = ['Outlet', 'Count']
            
            market_chart = alt.Chart(market_counts).mark_bar().encode(
                x='Count:Q',
                y='Outlet:N',
                color='Outlet:N',
                tooltip=['Outlet', 'Count']
            ).properties(
                title='Main Market Outlets',
                width=400,
                height=400
            )
            
            st.altair_chart(market_chart)
    
    with col2:
        # Income by farm size
        if '5.3 Total Income from Avocado Sales (KSH last season)' in df.columns and '2.1 Total Farm Size (Acres)' in df.columns:
            df['Farm Size Category'] = pd.cut(df['2.1 Total Farm Size (Acres)'],
                                            bins=[0, 3, 10, float('inf')],
                                            labels=['Small (<3 acres)', 'Medium (3-10 acres)', 'Large (>10 acres)'])
            
            income_df = df.groupby('Farm Size Category')['5.3 Total Income from Avocado Sales (KSH last season)'].mean().reset_index()
            
            income_chart = alt.Chart(income_df).mark_bar().encode(
                x='Farm Size Category:N',
                y='5.3 Total Income from Avocado Sales (KSH last season):Q',
                color='Farm Size Category:N',
                tooltip=['Farm Size Category', '5.3 Total Income from Avocado Sales (KSH last season)']
            ).properties(
                title='Average Income by Farm Size',
                width=400,
                height=400
            )
            
            st.altair_chart(income_chart)
    
    # Price analysis for Hass variety
    if '5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)' in df.columns:
        st.subheader("Hass Avocado Price Analysis")
        
        # Remove outliers
        hass_prices = df['5.2 Average Selling Price of (Hass variety) per kg last Season (KSH)'].dropna()
        hass_prices = hass_prices[hass_prices <= 120]  # Remove prices above 120
        
        if not hass_prices.empty:
            st.write(f"Average Price (outliers removed): Ksh {hass_prices.mean():.2f}")
            
            price_chart = alt.Chart(pd.DataFrame({'Price': hass_prices})).mark_bar().encode(
                alt.X("Price:Q", bin=True),
                y='count()',
                tooltip=['count()']
            ).properties(
                title='Distribution of Hass Avocado Prices (Ksh/kg)',
                width=600,
                height=400
            )
            
            st.altair_chart(price_chart)
    
    # Challenges word cloud
    if '5.10 Challenges in Market Access/Quality Standards' in df.columns:
        challenges_text = ""
        challenge_cols = [
            '5.10 Challenges in Market Access/Price Fluctuations',
            '5.10 Challenges in Market Access/Limited Buyers',
            '5.10 Challenges in Market Access/Quality Standards',
            '5.10 Challenges in Market Access/Other'
        ]
        
        for col in challenge_cols:
            if col in df.columns and df[col].sum() > 0:
                challenge_name = col.split('/')[-1]
                challenges_text += (challenge_name + ' ') * int(df[col].sum())
        
        if challenges_text:
            fig = create_wordcloud(challenges_text, "Market Access Challenges")
            st.pyplot(fig)

def show_training_needs(df):
    """Show training and extension needs"""
    st.subheader("Training & Extension Needs")
    
    if '8.6 What are your most pressing training/extension needs/GAP' in df.columns:
        needs_data = {
            'Need': ['GAP', 'Post-Harvest', 'Certification', 'Market Access'],
            'Count': [
                df['8.6 What are your most pressing training/extension needs/GAP'].sum(),
                df['8.6 What are your most pressing training/extension needs/Post-Harvest Management'].sum(),
                df['8.6 What are your most pressing training/extension needs/Certification Compliance'].sum(),
                df['8.6 What are your most pressing training/extension needs/Market Access'].sum()
            ]
        }
        
        needs_df = pd.DataFrame(needs_data)
        
        needs_chart = alt.Chart(needs_df).mark_bar().encode(
            x='Count:Q',
            y='Need:N',
            color='Need:N',
            tooltip=['Need', 'Count']
        ).properties(
            title='Most Pressing Training Needs',
            width=600,
            height=400
        )
        
        st.altair_chart(needs_chart)
    
    # Suggestions word cloud
    if 'Suggestions for the Shape Program Improvement' in df.columns:
        suggestions = ' '.join(df['Suggestions for the Shape Program Improvement'].dropna().astype(str))
        
        if suggestions.strip():
            fig = create_wordcloud(suggestions, "Farmer Suggestions for Program Improvement")
            st.pyplot(fig)

# --- Advanced Analysis Section ---
def show_advanced_analysis(df):
    """Show advanced analytical insights"""
    st.header("🔍 Analysis")
    
    if df.empty:
        st.warning("No data available for advanced analysis")
        return
    
    # Initialize analytics engine
    analytics = ShapeAdvancedAnalytics(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predictive Analytics", 
        "Farmer Segmentation", 
        "Correlation Analysis",
        "Economic Insights",
        "Certification Optimization"
    ])
    
    with tab1:
        st.subheader("Yield Prediction & Drivers")
        
        # Yield drivers analysis
        yield_analysis = analytics.analyze_yield_drivers()
        if yield_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Yield Prediction Accuracy (R²)", f"{yield_analysis['r2_score']:.3f}")
                fig = plot_feature_importance(yield_analysis['feature_importance'], 
                                           "Key Drivers of Avocado Yield")
                st.plotly_chart(fig)
            
            with col2:
                st.info("**Top Yield Drivers:**")
                for feature, importance in sorted(yield_analysis['feature_importance'].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"• {feature}: {importance:.3f}")
        else:
            st.warning("Insufficient data for yield prediction analysis")
    
    with tab2:
        st.subheader("Farmer Segmentation Analysis")
        
        # Farmer clustering
        segmentation = analytics.segment_farmers()
        if segmentation:
            # Add clusters to dataframe for display
            df_segmented = df.copy()
            df_segmented['Cluster'] = segmentation['clusters']
            df_segmented['Cluster_Label'] = [segmentation['cluster_labels'][c] for c in segmentation['clusters']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster distribution
                cluster_counts = df_segmented['Cluster_Label'].value_counts()
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                           title="Farmer Cluster Distribution")
                st.plotly_chart(fig)
            
            with col2:
                # Cluster profiles
                st.write("**Cluster Characteristics:**")
                for i, label in enumerate(segmentation['cluster_labels']):
                    st.write(f"**{label}:** {sum(segmentation['clusters'] == i)} farmers")
            
            # Detailed cluster analysis
            st.subheader("Cluster Profiles Analysis")
            fig = plot_cluster_profiles(segmentation, df)
            if fig:
                st.plotly_chart(fig)
        else:
            st.warning("Insufficient data for farmer segmentation")
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Comprehensive correlation analysis
        corr_data = analytics.correlation_analysis()
        if corr_data is not None:
            fig = plot_correlation_heatmap(corr_data, "Correlation Matrix - Key Metrics")
            st.plotly_chart(fig)
            
            # Highlight strong correlations
            strong_corrs = []
            for i in range(len(corr_data.columns)):
                for j in range(i+1, len(corr_data.columns)):
                    if abs(corr_data.iloc[i, j]) > 0.7:
                        strong_corrs.append((corr_data.columns[i], corr_data.columns[j], corr_data.iloc[i, j]))
            
            if strong_corrs:
                st.subheader("Strong Correlations Found")
                for corr in strong_corrs[:5]:  # Show top 5
                    st.write(f"**{corr[0]}** ↔ **{corr[1]}**: {corr[2]:.3f}")
        else:
            st.warning("Insufficient data for correlation analysis")
    
    with tab4:
        st.subheader("Economic Analysis")
        
        # Economic analysis
        economic_analysis = analytics.economic_analysis()
        if economic_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                # Income distribution
                fig = px.histogram(economic_analysis['income_per_acre'], 
                                 title="Income per Acre Distribution",
                                 labels={'value': 'Income per Acre (KSH)'})
                st.plotly_chart(fig)
            
            with col2:
                # Economic metrics
                st.metric("Average Income per Acre", f"Ksh {economic_analysis['stats']['mean']:,.0f}")
                st.metric("Median Income per Acre", f"Ksh {economic_analysis['stats']['median']:,.0f}")
                st.metric("Income Variability (Std Dev)", f"Ksh {economic_analysis['stats']['std']:,.0f}")
        else:
            st.warning("Insufficient data for economic analysis")
    
    with tab5:
        st.subheader("Certification Success Prediction")
        
        # Certification prediction
        cert_prediction = analytics.predict_certification_success()
        if cert_prediction:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Certification Prediction Accuracy", f"{cert_prediction['accuracy']:.3f}")
                fig = plot_feature_importance(cert_prediction['feature_importance'],
                                           "Key Drivers of Certification Success")
                st.plotly_chart(fig)
            
            with col2:
                st.info("**Top Certification Drivers:**")
                for feature, importance in sorted(cert_prediction['feature_importance'].items(),
                                               key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"• {feature}: {importance:.3f}")
                
                # Success probability distribution
                fig = px.histogram(cert_prediction['y_pred_proba'],
                                 title="Certification Success Probability Distribution",
                                 nbins=20)
                st.plotly_chart(fig)
        else:
            st.warning("Insufficient data for certification prediction")

def main():
    st.title("🥑 SHAPe Avocado Dashboard")
    st.markdown("Monitoring Kenya's avocado value chain for export excellence")
    
    # Load data with error handling
    try:
        farmer_df = load_farmer_data()
        metrics_df = load_exporter_metrics()
        
        if farmer_df.empty:
            st.warning("No farmer data loaded. Please check your data file.")
            return
    
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to:",
            ["📊 Monitoring Dashboard", "🔍 Analysis"]
        )
        
        # Sidebar filters - PERMANENTLY VISIBLE SECTION
        st.sidebar.title("Filters")
        
        # Company filter - permanently expanded
        st.sidebar.markdown("**Select Exporter**")
        companies = farmer_df['1.1 Company Name'].unique()
        selected_company = st.sidebar.radio(
            "Exporter:",
            options=['All'] + list(companies),
            index=0,
            key="exporter_radio"
        )
        
        # Date filter - permanently expanded
        st.sidebar.markdown("**Select Date Range**")
        if not farmer_df['submitdate'].isna().all():
            min_date = farmer_df['submitdate'].min().date()
            max_date = farmer_df['submitdate'].max().date()
            
            date_range = st.sidebar.date_input(
                "Date range:",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                key="date_range"
            )
        
        # Apply filters
        filtered_df = farmer_df.copy()
        
        if selected_company != 'All':
            filtered_df = filtered_df[filtered_df['1.1 Company Name'] == selected_company]
        
        if not farmer_df['submitdate'].isna().all() and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['submitdate'].dt.date >= date_range[0]) & 
                (filtered_df['submitdate'].dt.date <= date_range[1])
            ]
        
        # Display selected page
        if page == "📊 Monitoring Dashboard":
            show_overview(filtered_df, metrics_df)
            show_geospatial(filtered_df)
            show_certification(filtered_df)
            show_production_metrics(filtered_df)

            # --- Investor-oriented sections ---
            show_investor_income_view(filtered_df)
            show_income_potential_forecast(filtered_df)

            show_market_analysis(filtered_df)
            show_training_needs(filtered_df)
            
            # Data explorer
            st.subheader("Data Explorer")
            if st.checkbox("Show raw data"):
                st.dataframe(filtered_df)
            
            # Download button
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(filtered_df)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name='shape_filtered_data.csv',
                mime='text/csv'
            )
            
        else:  # Advanced Analytics page
            show_advanced_analysis(filtered_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
