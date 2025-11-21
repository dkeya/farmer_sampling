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
    .substitute-farm {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AvocadoSamplingApp:
    def __init__(self):
        # Removed Embu, increased to 4 farms per county, added substitute farms
        self.counties = {
            'MUR': {'name': 'Murang\'a', 'altitude_range': (1300, 1800), 'farms_needed': 4, 'substitutes': 2},
            'MER': {'name': 'Meru', 'altitude_range': (1500, 2000), 'farms_needed': 4, 'substitutes': 2},
            'NAK': {'name': 'Nakuru', 'altitude_range': (1800, 2300), 'farms_needed': 4, 'substitutes': 2},
            'UAS': {'name': 'Uasin Gishu', 'altitude_range': (2100, 2500), 'farms_needed': 4, 'substitutes': 2}
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
    
    def select_farmers_from_database(self, farmer_df):
        """Select appropriate farmers from SHAPe database for sampling"""
        selected_farms = []
        substitute_farms = []
        
        for county_code, county_info in self.counties.items():
            # Filter for county and Hass variety
            county_farms = farmer_df[
                (farmer_df['1.3 County'].str.contains(county_info['name'], na=False)) &
                (farmer_df['3.1 Variety Grown/Hass'] == 1)  # Hass variety only
            ].copy()
            
            if county_farms.empty:
                st.warning(f"No Hass avocado farms found in {county_info['name']}")
                continue
            
            # Filter by tree age (4-12 years) - using experience as proxy
            if '1.14 Experience in Avocado farming in years' in county_farms.columns:
                county_farms = county_farms[
                    (county_farms['1.14 Experience in Avocado farming in years'] >= 4) &
                    (county_farms['1.14 Experience in Avocado farming in years'] <= 12)
                ]
            
            # Separate large and small farms
            large_farms = county_farms[county_farms['2.1 Total Farm Size (Acres)'] > 10]
            small_farms = county_farms[county_farms['2.1 Total Farm Size (Acres)'] <= 3]
            
            # Select primary farms (2 large + 2 small per county)
            selected_large = large_farms.head(2)
            selected_small = small_farms.head(2)
            
            selected_county_farms = pd.concat([selected_large, selected_small])
            
            # Select substitute farms (1 large + 1 small per county)
            substitute_large = large_farms.iloc[2:3] if len(large_farms) > 2 else pd.DataFrame()
            substitute_small = small_farms.iloc[2:3] if len(small_farms) > 2 else pd.DataFrame()
            substitute_county_farms = pd.concat([substitute_large, substitute_small])
            
            # Add primary farms
            for idx, farm in selected_county_farms.iterrows():
                selected_farms.append({
                    'county_code': county_code,
                    'county_name': county_info['name'],
                    'farm_id': f"F{str(len(selected_farms) + 1).zfill(2)}",
                    'farmer_name': farm.get('1.10 Farmer\'s Name (Three Names)', 'N/A'),
                    'farm_size': farm.get('2.1 Total Farm Size (Acres)', 0),
                    'trees_planted': farm.get('2.3 Number of Avocado Trees Planted', 0),
                    'gps_lat': farm.get('_1.21 GPS Coordinates of Orchard_latitude', 'N/A'),
                    'gps_lon': farm.get('_1.21 GPS Coordinates of Orchard_longitude', 'N/A'),
                    'farm_type': 'Large' if farm.get('2.1 Total Farm Size (Acres)', 0) > 10 else 'Smallholder',
                    'experience_years': farm.get('1.14 Experience in Avocado farming in years', 'N/A'),
                    'status': 'Primary'
                })
            
            # Add substitute farms
            for idx, farm in substitute_county_farms.iterrows():
                substitute_farms.append({
                    'county_code': county_code,
                    'county_name': county_info['name'],
                    'farm_id': f"SUB{str(len(substitute_farms) + 1).zfill(2)}",
                    'farmer_name': farm.get('1.10 Farmer\'s Name (Three Names)', 'N/A'),
                    'farm_size': farm.get('2.1 Total Farm Size (Acres)', 0),
                    'trees_planted': farm.get('2.3 Number of Avocado Trees Planted', 0),
                    'gps_lat': farm.get('_1.21 GPS Coordinates of Orchard_latitude', 'N/A'),
                    'gps_lon': farm.get('_1.21 GPS Coordinates of Orchard_longitude', 'N/A'),
                    'farm_type': 'Large' if farm.get('2.1 Total Farm Size (Acres)', 0) > 10 else 'Smallholder',
                    'experience_years': farm.get('1.14 Experience in Avocado farming in years', 'N/A'),
                    'status': 'Substitute'
                })
        
        return selected_farms, substitute_farms
    
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
                    'county_name': farm['county_name'],
                    'farm_status': farm['status']
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
                        'lab_images_required': 4 if dm_testing else 0,
                        'farm_status': tree['farm_status']
                    })
        
        return sampling_schedule
    
    def calculate_sample_summary(self, sampling_schedule, selected_farms, substitute_farms):
        """Calculate comprehensive sample summary"""
        total_rounds = len(set([s['round'] for s in sampling_schedule]))
        total_primary_farms = len(selected_farms)
        total_substitute_farms = len(substitute_farms)
        total_trees = total_primary_farms * 6
        total_fruit_samples = len(sampling_schedule)
        
        dm_testing_samples = len([s for s in sampling_schedule if s['dm_testing_required']])
        
        # Image calculations
        on_tree_images = total_fruit_samples * 2
        lab_images = dm_testing_samples * 4
        total_images = on_tree_images + lab_images
        
        return {
            'Primary Farms': total_primary_farms,
            'Substitute Farms': total_substitute_farms,
            'Total Farms Available': total_primary_farms + total_substitute_farms,
            'Total Trees': total_trees,
            'Total Sampling Rounds': total_rounds,
            'Total Fruit Samples': total_fruit_samples,
            'DM Testing Samples': dm_testing_samples,
            'On-Tree Images': on_tree_images,
            'Lab Images': lab_images,
            'Total Images for AI': total_images,
            'Counties Covered': len(self.counties)
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
    **Updated Parameters:**
    - 4 counties (Embu removed)
    - 4 primary farms per county
    - 2 substitute farms per county
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
            
            # Generate sampling plan
            if st.button("🚀 Generate Sampling Plan", type="primary"):
                with st.spinner("Generating comprehensive sampling plan..."):
                    
                    # Step 1: Select farms
                    selected_farms, substitute_farms = app.select_farmers_from_database(farmer_df)
                    
                    # Step 2: Generate tree plan
                    tree_plan = app.generate_tree_sampling_plan(selected_farms)
                    
                    # Step 3: Generate sampling schedule
                    sampling_schedule = app.generate_fruit_sampling_schedule(tree_plan)
                    
                    # Step 4: Calculate summary
                    summary = app.calculate_sample_summary(sampling_schedule, selected_farms, substitute_farms)
                    
                    # Display results
                    st.markdown("## 📋 Sampling Plan Results")
                    
                    # Summary cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Primary Farms", summary['Primary Farms'])
                    with col2:
                        st.metric("Substitute Farms", summary['Substitute Farms'])
                    with col3:
                        st.metric("Total Trees", summary['Total Trees'])
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
                    st.markdown("### 🏠 Primary Farms")
                    primary_farms_df = pd.DataFrame(selected_farms)
                    st.dataframe(primary_farms_df)
                    
                    # Substitute farms table
                    st.markdown("### 🔄 Substitute Farms (Backup)")
                    if substitute_farms:
                        substitute_farms_df = pd.DataFrame(substitute_farms)
                        st.dataframe(substitute_farms_df)
                        
                        st.info("""
                        **Substitute Farm Protocol:**
                        - Keep substitute farmer contacts readily available
                        - If a primary farm drops out, immediately contact substitutes in the same county
                        - Maintain the same farm type balance (large/smallholder)
                        """)
                    else:
                        st.warning("No substitute farms available. Consider expanding farm selection criteria.")
                    
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
                        all_farms = selected_farms + substitute_farms
                        farms_df = pd.DataFrame(all_farms)
                        csv_farms = farms_df.to_csv(index=False)
                        st.download_button(
                            label="🏠 Download All Farms",
                            data=csv_farms,
                            file_name="all_farms_list.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        # Primary farms only
                        primary_df = pd.DataFrame(selected_farms)
                        csv_primary = primary_df.to_csv(index=False)
                        st.download_button(
                            label="🎯 Download Primary Farms",
                            data=csv_primary,
                            file_name="primary_farms.csv",
                            mime="text/csv"
                        )
                    
                    # Round-by-round sheets
                    st.markdown("#### 📅 Round-by-Round Collection Sheets")
                    round_cols = st.columns(4)
                    for round_num in range(1, 14):
                        round_data = schedule_df[schedule_df['round'] == round_num]
                        if not round_data.empty:
                            csv_round = round_data.to_csv(index=False)
                            with round_cols[(round_num-1) % 4]:
                                st.download_button(
                                    label=f"Round {round_num}",
                                    data=csv_round,
                                    file_name=f"round_{round_num}_collection_sheet.csv",
                                    mime="text/csv",
                                    key=f"round_{round_num}"
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
                    st.info(f"""
                    **Updated Plan Summary:**
                    - **{summary['Primary Farms']} primary farms** across {summary['Counties Covered']} counties
                    - **{summary['Substitute Farms']} substitute farms** for field contingencies
                    - **{summary['Total Fruit Samples']:,} total fruit samples** over 6 months
                    - **{summary['Total Images for AI']:,} images** for model training
                    
                    **Field Team Protocol:**
                    1. Begin with primary farms in all 4 counties
                    2. Keep substitute farmer contacts on hand during field visits
                    3. If a farm becomes unavailable, immediately activate same-county substitute
                    4. Maintain detailed logs of any farm substitutions
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
        3. **Click 'Generate Sampling Plan'** to create the sampling strategy
        4. **Download the generated plans** for field implementation
        
        ### 🎯 What you'll get:
        - **Primary farms** across 4 counties (Embu removed)
        - **Substitute farms** for field contingencies
        - **Complete sampling schedule** for 6 months
        - **Tree-by-tree sampling plan**
        - **Field collection sheets** for each round
        
        ### 📊 Updated Sample Requirements:
        - **4 counties** (Murang'a, Meru, Nakuru, Uasin Gishu)
        - **4 primary farms per county** (16 total)
        - **2 substitute farms per county** (8 total)
        - **6 trees per farm**
        - **5 fruits per tree**
        - **13 sampling rounds** over 6 months
        """)
        
        # Example of what the app does
        with st.expander("🔍 View Expected Output Structure"):
            st.markdown("""
            **Primary Farms Table:**
            - County, Farm ID, Farmer Name, Farm Size, Status (Primary)
            
            **Substitute Farms Table:**
            - County, Farm ID, Farmer Name, Farm Size, Status (Substitute)
            
            **Sampling Schedule:**
            - Round Number, Date, Farm ID, Tree ID, Fruit ID, DM Testing Requirement
            
            **Field Collection Sheets:**
            - Unique Fruit IDs, Sampling Instructions, Image Requirements
            """)

if __name__ == "__main__":
    main()