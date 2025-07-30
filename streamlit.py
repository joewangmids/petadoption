import streamlit as st
import pandas as pd
import io
import boto3
import altair as alt
from datetime import datetime

def load_data_from_s3(bucket, key):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets.aws.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=st.secrets.aws.AWS_SECRET_ACCESS_KEY,
            region_name=st.secrets.aws.AWS_REGION
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj['Body'])
    except Exception as e:
        st.error(f"Error loading data from S3: {e}")
        return None

# --- Configuration ---
S3_BUCKET_NAME = "xgb-los-multi"
FILE_KEY = "lz-multiclass/final_pipeline_prediction.csv"

df = load_data_from_s3(S3_BUCKET_NAME, FILE_KEY)

st.write(df.columns.T)


# --- 1. CONFIGURATION & HELPER FUNCTIONS ---

st.set_page_config(layout="wide")

# Initialize session state
if 'selected_animal_id' not in st.session_state:
    st.session_state.selected_animal_id = None

EMOJI_MAP = {
    "Breed": "🐕", "Age": "🎂", "Intake Type": "🏷️",
    "Sex": "🚻", "Has Name": "📛", "Color": "🎨"
}

def generate_full_dashboard_html(pet_data):
    predicted_proba = pet_data.get('predicted_proba', 0)

    # Create a formatted string for display
    formatted_proba = f"{(predicted_proba * 100):.2f}%"

    # Calculate the width for the progress bar
    progress_bar_width = raw_proba * 100
    
    animal_id = pet_data.get('animal_id', 'N/A')
    # raw_predicted_stay = pet_data.get('predicted_stay', 'N/A')
    recommended_team = pet_data.get('recommended_team', 'N/A')
    
    # if predicted_proba < 50: predicted_stay = "N/A"
    # else:
    #     try: predicted_stay = f"{int(raw_predicted_stay)}+ Days"
    #     except (ValueError, TypeError): predicted_stay = raw_predicted_stay

    factors_html = ""
    for i in range(1, 4):
        factor_name = pet_data.get(f'Positive_Feature_{i}', '')
        if not factor_name: continue
        factor_value = pet_data.get(f'Positive_Feature_{i}_Relative_Diff', '')
        raw_shap = pet_data.get(f'Positive_Feature_{i}_Relative_Diff', 0)
        more_or_less = "less" if raw_shap < 0 else "more"
        formatted_shap = f"{int(abs(raw_shap) * 100)}%"
        unit = " years" if factor_name == "Age" else ""
        statistic_string = f"{factor_value}{unit} are {formatted_shap} {more_or_less} likely to be adopted."
        emoji = EMOJI_MAP.get(f'Positive_Feature_{i}_Relative_Difff', '❓')
        factors_html += f"""<div class="flex items-center gap-2"><div class="text-xl text-gray-600">{emoji}</div><div><div class="font-medium text-sm">{factor_name}</div><div class="text-xs text-gray-600">{statistic_string}</div></div></div>"""

    team_html_module = "" # Default to an empty string
    if pd.notna(recommended_team):
        team_html_module = f"""
        <div class="team-section">
            <div class="team-header">
                <div class="team-avatar"><i class="fas fa-hands-helping"></i></div>
                <div class="team-info">
                    <h3>{recommended_team}</h3>
                    <div class="team-title">Recommended Team</div>
                </div>
            </div>
        </div>
        """

    if predicted_proba * 100 < 25: progress_color, risk_category = "bg-red-500", "High Risk"
    elif predicted_proba * 100 < 50: progress_color, risk_category = "bg-yellow-500", "Medium Risk"
    else: progress_color, risk_category = "bg-green-500", "Low Risk"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {{ font-family:'Inter', sans-serif; }}
            .module {{ padding: 1rem; }}
            .progress-bar {{ height:8px; border-radius:4px; background-color:#e9ecef; }}
            .progress-fill {{ height:100%; border-radius:4px; }}
            .team-section {{ background-color: #f8fafc; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }}
            .team-header {{ display: flex; align-items: center; gap: 0.75rem; }}
            .team-avatar {{ background-color: #e0f2fe; padding: 0.75rem; border-radius: 9999px; }}
            .team-avatar i {{ color: #0ea5e9; }}
            .team-info h3 {{ font-weight: 600; font-size: 0.875rem; line-height: 1.25rem; }}
            .team-info .team-title {{ font-size: 0.75rem; line-height: 1rem; color: #64748b; }}
        </style>
    </head>
    <body>
        <div class="bg-white p-4 sm:p-6">
            <div class="mb-3">
                <h1 class="text-2xl font-bold text-gray-800">Pet Adoptability Dashboard</h1>
                <p class="text-sm text-gray-500">Pet ID: #{animal_id}</p>
            </div>
            <div class="flex flex-col gap-4 max-w-3xl mx-auto">
                <div class="bg-gray-50 rounded-lg p-4 shadow-sm module">
                    <h2 class="text-lg font-bold text-gray-700 mb-2">Adoption Score</h2>
                    <h3 class="font-bold text-gray-700">Score: {formatted_proba}%</h3>
                    <div class="progress-bar mt-1"><div class="progress-fill {progress_color}" style="width:{formatted_proba}%"></div></div>
                    <div class="mt-3"><span class="{progress_color} text-white px-3 py-0.5 rounded-full text-sm font-medium">{risk_category}</span></div>
                </div>
                {team_html_module}
                <div class="bg-gray-50 rounded-lg p-4 shadow-sm module">
                    <h2 class="text-lg font-bold text-gray-700 mb-2">Top Factors Affecting Adoption Score</h2>
                    <div class="space-y-2">{factors_html}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def color_predicted_proba(val):
    """Color predicted_probas based on risk level"""
    if val < 25:
        return 'background-color: #FF6B6B; color: white'
    elif val < 50:
        return 'background-color: #FFD166'
    else:
        return 'background-color: #06D6A0; color: white'

# --- 2. MAIN APP WORKFLOW ---

st.title("🐾 Shelter Pet Priority Board")
st.write("This board automatically surfaces the pets that need the most attention first.")
st.markdown("<p style='color: red;'>**Note: 'High Risk' means a pet is at a high risk of NOT being adopted**</p>", unsafe_allow_html=True)

if df is not None:
    
    st.sidebar.header("Filter Options")

    if 'intake_date' in df.columns:
        df['intake_date'] = pd.to_datetime(df['intake_date'], errors='coerce')

    # Create a copy of the dataframe to filter
    filtered_df = df.copy()

    # 1. Animal Type Filter
    animal_types = sorted(df['animal_type'].dropna().unique())
    selected_animal_types = st.sidebar.multiselect(
        'Filter by Animal Type:',
        options=animal_types,
        default=animal_types # Default to all types selected
    )

    # Apply filters sequentially
    if selected_animal_types:
        filtered_df = filtered_df[filtered_df['animal_type'].isin(selected_animal_types)]
    
    # --- DATA PREPARATION FOR SUMMARY DASHBOARD ---
    
    # 1. Create adoptability category column
    def get_adoptability_category(predicted_proba):
        if predicted_proba <= 25: return "High Risk"
        if predicted_proba < 50: return "Medium Risk"
        return "Low Risk"
    filtered_df['adoptability_category'] = filtered_df['predicted_proba'].apply(get_adoptability_category) 

    # 2. Create predicted stay bins column
    def get_stay_bin(stay):
        if pd.isna(stay) or str(stay).lower() == 'n/a': return "N/A"
        try:
            stay = int(stay)
            if stay <= 30: return "0-30 Days"
            if stay <= 90: return "31-90 Days"
            return "90+ Days"
        except (ValueError, TypeError):
            return "N/A"
    # filtered_df['stay_bin'] = filtered_df['predicted_stay'].apply(get_stay_bin) 

    # --- SUMMARY DASHBOARD ---
    with st.expander("Show Shelter-Wide Summary Dashboard", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Pets by Adoptability")
            category_chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X('count():Q', title="Number of Pets"),
                y=alt.Y('adoptability_category:N', title="Category", sort=['Low Risk', 'Medium Risk', 'High Risk']),
                color=alt.Color('adoptability_category:N', 
                                scale=alt.Scale(domain=['High Risk', 'Medium Risk', 'Low Risk'], 
                                                range=['#FF6B6B', '#FFD166', '#06D6A0']),
                                legend=None)
            ).properties(height=200)
            st.altair_chart(category_chart, use_container_width=True)

        with col2:
            st.subheader("Pets by Predicted Stay")
            stay_chart = alt.Chart(filtered_df).mark_bar().encode(
                x=alt.X('count():Q', title="Number of Pets"),
                y=alt.Y('stay_bin:N', title="Predicted Stay", sort=['0-30 Days', '31-90 Days', '90+ Days', 'N/A'])
            ).properties(height=200)
            st.altair_chart(stay_chart, use_container_width=True)

        st.subheader("Distribution of Adoption predicted_probas")
        predicted_proba_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("predicted_proba:Q", bin=alt.Bin(maxbins=20), title="Adoption predicted_proba"),
            alt.Y('count():Q', title="Number of Pets"),
        ).properties(height=250)
        st.altair_chart(predicted_proba_hist, use_container_width=True)

    # --- TRIAGE BOARD DISPLAY ---
    sorted_df = filtered_df.sort_values(by="predicted_proba", ascending=True)
    
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("Triage List")

        # --- LEGEND ---
        st.markdown("""
            <style>
                .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
                .legend-color { width: 15px; height: 15px; margin-right: 8px; border-radius: 3px; }
            </style>
            <b>Adoption Score Legend:</b>
            <p style='color: red;'>**Note: 'High Risk' means a pet is at a high risk of NOT being adopted**</p>
            """, unsafe_allow_html=True)

        leg1, leg2, leg3 = st.columns(3)
        with leg1:
            st.markdown("<div class='legend-item'><div class='legend-color' style='background-color:#FF6B6B;'></div> High Risk (< 25)</div>", unsafe_allow_html=True)
        with leg2:
            st.markdown("<div class='legend-item'><div class='legend-color' style='background-color:#FFD166;'></div> Medium Risk (25-49)</div>", unsafe_allow_html=True)
        with leg3:
            st.markdown("<div class='legend-item'><div class='legend-color' style='background-color:#06D6A0;'></div> Low Risk (≥ 50)</div>", unsafe_allow_html=True)
        
        st.write("Click on a row to view pet details")
        
        df_display = sorted_df[['animal_id', 'predicted_proba', 'Positive_Feature_1']].rename(columns={
            'animal_id': 'Pet ID', 
            'predicted_proba': 'predicted_proba', 
            'Positive_Feature_1': 'Primary Concern'
        }).reset_index(drop=True)
        
        try:
            event = st.dataframe(
                df_display.style.applymap(color_predicted_proba, subset=['predicted_proba']).format({'predicted_proba': '{:.2%}'}),
                use_container_width=True,
                height=400,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if event.selection and len(event.selection.rows) > 0:
                selected_idx = event.selection.rows[0]
                selected_animal_id = df_display.iloc[selected_idx]['Pet ID']
                if st.session_state.selected_animal_id != selected_animal_id:
                    st.session_state.selected_animal_id = selected_animal_id
                    st.rerun()
                    
        except Exception as e:
            st.dataframe(
                df_display.style.applymap(color_predicted_proba, subset=['predicted_proba']),
                use_container_width=True,
                height=300
            )
            if len(df_display) > 0:
                pet_options = [f"Pet {row['Pet ID']} - Probability: {(row['predicted_proba'] * 100):.1f}% - {row['Primary Concern']}" for _, row in df_display.iterrows()]
                selected_option = st.radio(
                    "Select a pet:",
                    options=pet_options,
                    index=0
                )
                selected_animal_id = int(selected_option.split(" ")[1])
                if st.session_state.selected_animal_id != selected_animal_id:
                    st.session_state.selected_animal_id = selected_animal_id
                    st.rerun()

    with col2:
        st.header("Pet Details")
        if st.session_state.selected_animal_id:
            pet_rows = sorted_df[sorted_df['animal_id'] == st.session_state.selected_animal_id]
            if not pet_rows.empty:
                selected_pet_data = pet_rows.iloc[0]
                full_detail_html = generate_full_dashboard_html(selected_pet_data)
                st.components.v1.html(full_detail_html, height=700, scrolling=False)
            else:
                st.info("Selected pet not found in filtered data. Please clear filters or select another pet.")
        else:
            st.info("Click on a row to view pet details.")

else:
    st.warning("Data could not be loaded. Please check S3 configuration and credentials.")
