import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(
    page_title="Avocado Dry Matter Sampling",
    page_icon="🥑",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin-bottom: 1rem;
    }
    .download-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .diagnostic-section {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AvocadoSamplingApp:
    def __init__(self):
        self.counties = {
            'MUR': {'name': 'Murang\'a', 'altitude_range': (1300, 1800), 'farms_needed': 3},
            'EMB': {'name': 'Embu', 'altitude_range': (1100, 1700), 'farms_needed': 3},
            'MER': {'name': 'Meru', 'altitude_range': (1500, 2000), 'farms_needed': 3},
            'NAK': {'name': 'Nakuru', 'altitude_range': (1800, 2300), 'farms_needed': 3},
            'UAS': {'name': 'Uasin Gishu', 'altitude_range': (2100, 2500), 'farms_needed': 3}
        }
        
    def generate_sampling_calendar(self):
        """Generate bi-weekly sampling schedule from Nov 2025 to Apr 2026"""
        start_date = datetime(2025, 11, 1)
        end_date = datetime(2026, 4, 30)
        current_date = start_date
        rounds = []
        
        round_num = 1
        while current_date <= end_date and round_num <= 13:
            rounds.append({
                'round': round_num,
                'date': current_date.strftime('%Y-%m-%d'),
                'week': f"Week {(current_date - start_date).days // 7 + 1}"
            })
            current_date += timedelta(days=14)
            round_num += 1
            
        return rounds
    
    def diagnose_county_issues(self, farmer_df, county_name="Embu"):
        """Diagnose why farms from a specific county aren't qualifying"""
        
        st.markdown(f'<div class="diagnostic-section">🔍 Diagnostic Check: {county_name} County</div>', unsafe_allow_html=True)
        
        # Check if county exists in data
        county_data = farmer_df[farmer_df['1.3 County'].str.contains(county_name, na=False)]
        
        if county_data.empty:
            st.error(f"❌ No farms found in {county_name} county")
            st.info("Check if the county name is spelled correctly in your data. Look for variations like 'Embu County', 'EMBU', etc.")
            return
        
        st.success(f"✅ Found {len(county_data)} farms in {county_name} county")
        
        # Display sample of county data
        with st.expander(f"View {county_name} farms data"):
            st.dataframe(county_data[['1.10 Farmer\'s Name (Three Names)', '1.3 County', '2.1 Total Farm Size (Acres)', '3.1 Variety Grown/Hass']].head())
        
        # Check Hass variety requirement
        if '3.1 Variety Grown/Hass' in county_data.columns:
            hass_farms = county_data[county_data['3.1 Variety Grown/Hass'] == 1]
            st.write(f"**Hass Variety Farms:** {len(hass_farms)} out of {len(county_data)}")
            
            if len(hass_farms) == 0:
                st.error("❌ No Hass avocado farms in this county")
                st.info("All farms must grow Hass variety. Check if '3.1 Variety Grown/Hass' column has value 1 for Hass growers.")
                return hass_farms
        else:
            st.warning("⚠️ '3.1 Variety Grown/Hass' column not found in data")
            hass_farms = county_data
        
        # Check farm size distribution
        if '2.1 Total Farm Size (Acres)' in hass_farms.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                large_farms = hass_farms[hass_farms['2.1 Total Farm Size (Acres)'] > 10]
                st.metric("Large Farms (>10 acres)", len(large_farms))
                if len(large_farms) == 0:
                    st.warning("Need at least 1 large farm")
            
            with col2:
                small_farms = hass_farms[hass_farms['2.1 Total Farm Size (Acres)'] <= 3]
                st.metric("Smallholder Farms (≤3 acres)", len(small_farms))
                if len(small_farms) < 2:
                    st.warning("Need at least 2 smallholder farms")
            
            with col3:
                medium_farms = hass_farms[
                    (hass_farms['2.1 Total Farm Size (Acres)'] > 3) & 
                    (hass_farms['2.1 Total Farm Size (Acres)'] <= 10)
                ]
                st.metric("Medium Farms", len(medium_farms))
            
            # Show farm sizes
            st.write("**Farm Size Distribution:**")
            st.dataframe(hass_farms[['1.10 Farmer\'s Name (Three Names)', '2.1 Total Farm Size (Acres)']].head(10))
        else:
            st.error("❌ '2.1 Total Farm Size (Acres)' column not found")
        
        # Check experience/tree age
        if '1.14 Experience in Avocado farming in years' in hass_farms.columns:
            suitable_age = hass_farms[
                (hass_farms['1.14 Experience in Avocado farming in years'] >= 4) &
                (hass_farms['1.14 Experience in Avocado farming in years'] <= 12)
            ]
            st.write(f"**Suitable Age (4-12 years):** {len(suitable_age)} out of {len(hass_farms)} farms")
            
            if len(suitable_age) == 0:
                st.error("❌ No farms with suitable tree age (4-12 years)")
                st.info("We use farming experience as a proxy for tree age. Farms need 4-12 years of avocado farming experience.")
        
        return hass_farms

    def select_farmers_from_database(self, farmer_df):
        """Select appropriate farmers from SHAPe database for sampling - ROBUST VERSION"""
        selected_farms = []
        
        # First, let's see what counties we actually have
        available_counties = farmer_df['1.3 County'].unique() if '1.3 County' in farmer_df.columns else []
        st.sidebar.write("**Available Counties:**", list(available_counties))
        
        for county_code, county_info in self.counties.items():
            county_name = county_info['name']
            
            # Flexible county name matching
            county_farms = farmer_df[
                farmer_df['1.3 County'].str.contains(county_name, na=False, case=False)
            ].copy()
            
            if county_farms.empty:
                st.warning(f"❌ No farms found for {county_name}")
                continue
            
            # Check Hass variety (handle different column scenarios)
            hass_column = '3.1 Variety Grown/Hass'
            if hass_column in county_farms.columns:
                # Convert to numeric if needed
                county_farms[hass_column] = pd.to_numeric(county_farms[hass_column], errors='coerce')
                hass_farms = county_farms[county_farms[hass_column] == 1]
                
                if hass_farms.empty:
                    st.warning(f"⚠️ No Hass avocado farms in {county_name}")
                    continue
            else:
                st.warning(f"⚠️ Hass variety column not found in {county_name}, using all farms")
                hass_farms = county_farms
            
            # Filter by experience/tree age if available
            experience_col = '1.14 Experience in Avocado farming in years'
            if experience_col in hass_farms.columns:
                hass_farms[experience_col] = pd.to_numeric(hass_farms[experience_col], errors='coerce')
                suitable_farms = hass_farms[
                    (hass_farms[experience_col] >= 4) & 
                    (hass_farms[experience_col] <= 12)
                ]
                if not suitable_farms.empty:
                    hass_farms = suitable_farms
            
            # Ensure farm size is numeric
            size_col = '2.1 Total Farm Size (Acres)'
            if size_col in hass_farms.columns:
                hass_farms[size_col] = pd.to_numeric(hass_farms[size_col], errors='coerce')
            
            # Select farms (with fallbacks)
            large_farms = hass_farms[hass_farms[size_col] > 10] if size_col in hass_farms.columns else pd.DataFrame()
            small_farms = hass_farms[hass_farms[size_col] <= 3] if size_col in hass_farms.columns else hass_farms.head(2)
            
            # If no large farms, use largest available
            if large_farms.empty and not hass_farms.empty:
                large_farms = hass_farms.nlargest(1, size_col) if size_col in hass_farms.columns else hass_farms.head(1)
            
            selected_large = large_farms.head(1)
            selected_small = small_farms.head(2)
            
            selected_county_farms = pd.concat([selected_large, selected_small]).drop_duplicates()
            
            for idx, farm in selected_county_farms.iterrows():
                selected_farms.append({
                    'county_code': county_code,
                    'county_name': county_name,
                    'farm_id': f"F{str(len(selected_farms) + 1).zfill(2)}",
                    'farmer_name': farm.get('1.10 Farmer\'s Name (Three Names)', 'N/A'),
                    'farm_size': farm.get(size_col, 0),
                    'trees_planted': farm.get('2.3 Number of Avocado Trees Planted', 0),
                    'gps_lat': farm.get('_1.21 GPS Coordinates of Orchard_latitude', 'N/A'),
                    'gps_lon': farm.get('_1.21 GPS Coordinates of Orchard_longitude', 'N/A'),
                    'farm_type': 'Large' if farm.get(size_col, 0) > 10 else 'Smallholder',
                    'experience_years': farm.get(experience_col, 'N/A')
                })
        
        return selected_farms
    
    def generate_tree_sampling_plan(self, selected_farms):
        """Generate detailed tree sampling plan for each farm"""
        tree_plan = []
        
        for farm in selected_farms:
            farm_id = farm['farm_id']
            county_code = farm['county_code']
            
            # 6 trees per farm with different positions
            tree_positions = ['Edge_North', 'Edge_South', 'Mid_Block_East', 
                            'Mid_Block_West', 'Interior_Center', 'Interior_Shaded']
            
            for i, position in enumerate(tree_positions, 1):
                tree_plan.append({
                    'county_code': county_code,
                    'farm_id': farm_id,
                    'tree_id': f"T{str(i).zfill(2)}",
                    'tree_number': i,
                    'position': position,
                    'unique_tree_code': f"{county_code}-{farm_id}-T{str(i).zfill(2)}",
                    'farmer_name': farm['farmer_name'],
                    'county_name': farm['county_name']
                })
        
        return tree_plan
    
    def generate_fruit_sampling_schedule(self, tree_plan):
        """Generate complete fruit sampling schedule for all rounds"""
        sampling_schedule = []
        sampling_calendar = self.generate_sampling_calendar()
        
        for round_info in sampling_calendar:
            round_num = round_info['round']
            sampling_date = round_info['date']
            
            for tree in tree_plan:
                # 5 fruits per tree per round
                for fruit_num in range(1, 6):
                    fruit_position = ['North', 'East', 'South', 'West', 'Center'][fruit_num - 1]
                    
                    # Determine if fruit goes for DM testing (3 per tree normally, 5 during peak)
                    dm_testing = True if fruit_num <= 3 else (round_num >= 10)  # Peak harvest rounds
                    
                    sampling_schedule.append({
                        'round': round_num,
                        'sampling_date': sampling_date,
                        'county_code': tree['county_code'],
                        'county_name': tree['county_name'],
                        'farm_id': tree['farm_id'],
                        'tree_id': tree['tree_id'],
                        'tree_position': tree['position'],
                        'farmer_name': tree['farmer_name'],
                        'fruit_number': fruit_num,
                        'fruit_position': fruit_position,
                        'dm_testing_required': dm_testing,
                        'unique_fruit_id': f"{tree['county_code']}-{tree['farm_id']}-{tree['tree_id']}-FR{str(fruit_num).zfill(2)}-R{str(round_num).zfill(2)}",
                        'on_tree_images_required': 2,
                        'lab_images_required': 4 if dm_testing else 0
                    })
        
        return sampling_schedule
    
    def calculate_sample_summary(self, sampling_schedule, selected_farms):
        """Calculate comprehensive sample summary"""
        total_rounds = len(set([s['round'] for s in sampling_schedule]))
        total_farms = len(selected_farms)
        total_trees = total_farms * 6
        total_fruit_samples = len(sampling_schedule)
        
        dm_testing_samples = len([s for s in sampling_schedule if s['dm_testing_required']])
        
        # Image calculations
        on_tree_images = total_fruit_samples * 2
        lab_images = dm_testing_samples * 4
        total_images = on_tree_images + lab_images
        
        return {
            'Total Farms': total_farms,
            'Total Trees': total_trees,
            'Total Sampling Rounds': total_rounds,
            'Total Fruit Samples': total_fruit_samples,
            'DM Testing Samples': dm_testing_samples,
            'On-Tree Images': on_tree_images,
            'Lab Images': lab_images,
            'Total Images for AI': total_images,
            'Smart Glasses Pilot Farms': min(10, total_farms)
        }

def main():
    st.markdown('<div class="main-header">🥑 Avocado Dry Matter Sampling Planner</div>', unsafe_allow_html=True)
    
    # Initialize app
    app = AvocadoSamplingApp()
    
    # Sidebar for file upload and parameters
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload SHAPe Baseline Data (Excel)", 
        type=['xlsx'],
        help="Upload the Baseline sheet from your SHAPe data"
    )
    
    st.sidebar.title("⚙️ Sampling Parameters")
    st.sidebar.info("""
    **Default Parameters:**
    - 5 counties
    - 15 farms total
    - 6 trees per farm
    - 5 fruits per tree
    - 13 sampling rounds
    """)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            farmer_df = pd.read_excel(uploaded_file, sheet_name='Baseline')
            st.success(f"✅ Successfully loaded data with {len(farmer_df)} farmer records")
            
            # Add diagnostic section after data loading
            st.sidebar.markdown("---")
            st.sidebar.title("🔍 County Diagnostics")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Check Embu Issues"):
                    app.diagnose_county_issues(farmer_df, "Embu")
            with col2:
                if st.button("Check All Counties"):
                    for county in ["Murang'a", "Embu", "Meru", "Nakuru", "Uasin Gishu"]:
                        app.diagnose_county_issues(farmer_df, county)
            
            # Display data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(farmer_df.head())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Farmers", len(farmer_df))
                with col2:
                    hass_farmers = farmer_df['3.1 Variety Grown/Hass'].sum() if '3.1 Variety Grown/Hass' in farmer_df.columns else "N/A"
                    st.metric("Hass Variety Farmers", hass_farmers)
                with col3:
                    counties = farmer_df['1.3 County'].nunique() if '1.3 County' in farmer_df.columns else "N/A"
                    st.metric("Counties Represented", counties)
            
            # Display available counties in data
            if '1.3 County' in farmer_df.columns:
                available_counties = farmer_df['1.3 County'].unique()
                with st.expander("🌍 Counties in Your Data"):
                    st.write("Counties found in your dataset:")
                    for county in available_counties:
                        st.write(f"- {county}")
            
            # Generate sampling plan
            if st.button("🚀 Generate Sampling Plan", type="primary"):
                with st.spinner("Generating comprehensive sampling plan..."):
                    
                    # Step 1: Select farms
                    selected_farms = app.select_farmers_from_database(farmer_df)
                    
                    # Step 2: Generate tree plan
                    tree_plan = app.generate_tree_sampling_plan(selected_farms)
                    
                    # Step 3: Generate sampling schedule
                    sampling_schedule = app.generate_fruit_sampling_schedule(tree_plan)
                    
                    # Step 4: Calculate summary
                    summary = app.calculate_sample_summary(sampling_schedule, selected_farms)
                    
                    # Display results
                    st.markdown("## 📋 Sampling Plan Results")
                    
                    # Summary cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Farms Selected", summary['Total Farms'])
                    with col2:
                        st.metric("Total Trees", summary['Total Trees'])
                    with col3:
                        st.metric("Sampling Rounds", summary['Total Sampling Rounds'])
                    with col4:
                        st.metric("Total Samples", summary['Total Fruit Samples'])
                    
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("DM Tests", summary['DM Testing Samples'])
                    with col6:
                        st.metric("On-Tree Images", f"{summary['On-Tree Images']:,}")
                    with col7:
                        st.metric("Lab Images", f"{summary['Lab Images']:,}")
                    with col8:
                        st.metric("Total Images", f"{summary['Total Images for AI']:,}")
                    
                    # Selected farms table
                    st.markdown("### 🏠 Selected Farms")
                    farms_df = pd.DataFrame(selected_farms)
                    st.dataframe(farms_df)
                    
                    # Sampling schedule by round
                    st.markdown("### 📅 Sampling Schedule Overview")
                    schedule_df = pd.DataFrame(sampling_schedule)
                    
                    # Round summary
                    round_summary = schedule_df.groupby('round').agg({
                        'unique_fruit_id': 'count',
                        'dm_testing_required': 'sum',
                        'on_tree_images_required': 'sum',
                        'lab_images_required': 'sum'
                    }).reset_index()
                    round_summary.columns = ['Round', 'Total Fruits', 'DM Tests', 'On-Tree Images', 'Lab Images']
                    
                    st.dataframe(round_summary)
                    
                    # Download section
                    st.markdown("---")
                    st.markdown("## 📥 Download Sampling Plan")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Complete sampling plan
                        csv_schedule = schedule_df.to_csv(index=False)
                        st.download_button(
                            label="📋 Download Complete Schedule",
                            data=csv_schedule,
                            file_name="complete_sampling_schedule.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Selected farms list
                        csv_farms = farms_df.to_csv(index=False)
                        st.download_button(
                            label="🏠 Download Selected Farms",
                            data=csv_farms,
                            file_name="selected_farms.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # Round-by-round sheets
                        for round_num in range(1, 14):
                            round_data = schedule_df[schedule_df['round'] == round_num]
                            if not round_data.empty:
                                csv_round = round_data.to_csv(index=False)
                                st.download_button(
                                    label=f"📅 Round {round_num} Sheet",
                                    data=csv_round,
                                    file_name=f"round_{round_num}_collection_sheet.csv",
                                    mime="text/csv"
                                )
                    
                    # Tree plan download
                    st.markdown("### 🌳 Tree Sampling Plan")
                    tree_df = pd.DataFrame(tree_plan)
                    csv_trees = tree_df.to_csv(index=False)
                    st.download_button(
                        label="🌳 Download Tree Plan",
                        data=csv_trees,
                        file_name="tree_sampling_plan.csv",
                        mime="text/csv"
                    )
                    
                    # Instructions
                    st.markdown("---")
                    st.markdown("## 🎯 Implementation Instructions")
                    st.info("""
                    **Next Steps:**
                    1. **Review selected farms** with county agricultural officers
                    2. **Schedule farmer orientation meetings** for November 2025
                    3. **Procure equipment**: Smart glasses, cameras, sampling bags
                    4. **Train field team** on data collection protocols
                    5. **Begin sampling** according to the generated schedule
                    """)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure you're uploading the correct Excel file with the 'Baseline' sheet")
    
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Avocado Dry Matter Sampling Planner!
        
        This tool will help you select farms and generate a comprehensive sampling plan for the AI Dry Matter prediction model.
        
        ### 📋 How to use:
        1. **Upload your SHAPe Baseline data** (Excel file) in the sidebar
        2. **Review the data preview** to ensure correct loading
        3. **Use the diagnostic tools** to check county eligibility
        4. **Click 'Generate Sampling Plan'** to create the sampling strategy
        5. **Download the generated plans** for field implementation
        
        ### 🎯 What you'll get:
        - **Selected farms** across 5 counties
        - **Complete sampling schedule** for 6 months
        - **Tree-by-tree sampling plan**
        - **Field collection sheets** for each round
        - **Smart glasses pilot** farm selection
        
        ### 📊 Sample Requirements:
        - **15 farms** (3 per county)
        - **90 trees** (6 per farm)
        - **5,850+ fruit samples**
        - **27,540+ images** for AI training
        - **13 sampling rounds** over 6 months
        """)
        
        # Example of what the app does
        with st.expander("🔍 View Expected Output Structure"):
            st.markdown("""
            **Selected Farms Table:**
            - County, Farm ID, Farmer Name, Farm Size, GPS Coordinates
            
            **Sampling Schedule:**
            - Round Number, Date, Farm ID, Tree ID, Fruit ID, DM Testing Requirement
            
            **Field Collection Sheets:**
            - Unique Fruit IDs, Sampling Instructions, Image Requirements
            """)

if __name__ == "__main__":
    main()