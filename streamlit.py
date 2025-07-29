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
S3_BUCKET_NAME = "vizpickuplocation"
FILE_KEY = "viz_landing_zone/pets_data.csv"

# --- 1. CONFIGURATION & HELPER FUNCTIONS ---

st.set_page_config(layout="wide")

# Initialize session state
if 'selected_pet_id' not in st.session_state:
    st.session_state.selected_pet_id = None

EMOJI_MAP = {
    "Breed": "🐕", "Age": "🎂", "Intake Type": "🏷️",
    "Sex": "🚻", "Has Name": "📛", "Color": "🎨"
}

def generate_full_dashboard_html(pet_data):
    """This is your detailed dashboard function from the previous step."""
    score = pet_data.get('score', 0)
    pet_id = pet_data.get('pet_id', 'N/A')
    raw_predicted_stay = pet_data.get('predicted_stay', 'N/A')
    recommended_team = pet_data.get('recommended_team', 'N/A')
    
    if score < 50: predicted_stay = "N/A"
    else:
        try: predicted_stay = f"{int(raw_predicted_stay)}+ Days"
        except (ValueError, TypeError): predicted_stay = raw_predicted_stay

    factors_html = ""
    for i in range(1, 4):
        factor_name = pet_data.get(f'factor_{i}_name', '')
        if not factor_name: continue
        factor_value = pet_data.get(f'factor_{i}_value', '')
        raw_shap = pet_data.get(f'factor_{i}_shap', 0)
        more_or_less = "less" if raw_shap < 0 else "more"
        formatted_shap = f"{int(abs(raw_shap) * 100)}%"
        unit = " years" if factor_name == "Age" else ""
        statistic_string = f"{factor_value}{unit} are {formatted_shap} {more_or_less} likely to be adopted."
        emoji = EMOJI_MAP.get(factor_name, '❓')
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

    if score < 25: progress_color, risk_category = "bg-red-500", "High Risk"
    elif score < 50: progress_color, risk_category = "bg-yellow-500", "Medium Risk"
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
                <p class="text-sm text-gray-500">Pet ID: #{pet_id}</p>
            </div>
            <div class="flex flex-col gap-4 max-w-3xl mx-auto">
                <div class="bg-gray-50 rounded-lg p-4 shadow-sm module">
                    <h2 class="text-lg font-bold text-gray-700 mb-2">Adoption Score</h2>
                    <h3 class="font-bold text-gray-700">Score: {score}</h3>
                    <div class="progress-bar mt-1"><div class="progress-fill {progress_color}" style="width:{score}%"></div></div>
                    <div class="mt-3"><span class="{progress_color} text-white px-3 py-0.5 rounded-full text-sm font-medium">{risk_category}</span></div>
                    <div class="mt-3 flex items-center gap-2 text-sm text-gray-700"><i class="fas fa-calendar-alt text-gray-500"></i><span>Predicted Stay: {predicted_stay}</span></div>
                </div>
                {team_html_module}
                <div class="bg-gray-50 rounded-lg p-4 shadow-sm module">
                    <h2 class="text-lg font-bold text-gray-700 mb-2">Top Factors Affecting Score</h2>
                    <div class="space-y-2">{factors_html}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def color_score(val):
    """Color scores based on risk level"""
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

# Load the data
df = load_data_from_s3(S3_BUCKET_NAME, FILE_KEY)

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
    def get_adoptability_category(score):
        if score <= 33: return "High Risk"
        if score <= 66: return "Medium Risk"
        return "Low Risk"
    filtered_df['adoptability_category'] = filtered_df['score'].apply(get_adoptability_category) 

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
    filtered_df['stay_bin'] = filtered_df['predicted_stay'].apply(get_stay_bin) 

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

        st.subheader("Distribution of Adoption Scores")
        score_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Adoption Score"),
            alt.Y('count():Q', title="Number of Pets"),
        ).properties(height=250)
        st.altair_chart(score_hist, use_container_width=True)

    # --- TRIAGE BOARD DISPLAY ---
    sorted_df = filtered_df.sort_values(by="score", ascending=True)
    
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("Triage List")

        # --- LEGEND ---
        st.markdown("""
            <style>
                .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
                .legend-color { width: 15px; height: 15px; margin-right: 8px; border-radius: 3px; }
            </style>
            <b>Score Legend:</b>
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
        
        df_display = sorted_df[['pet_id', 'score', 'factor_1_name']].rename(columns={
            'pet_id': 'Pet ID', 
            'score': 'Score', 
            'factor_1_name': 'Primary Concern'
        }).reset_index(drop=True)
        
        try:
            event = st.dataframe(
                df_display.style.applymap(color_score, subset=['Score']),
                use_container_width=True,
                height=400,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if event.selection and len(event.selection.rows) > 0:
                selected_idx = event.selection.rows[0]
                selected_pet_id = df_display.iloc[selected_idx]['Pet ID']
                if st.session_state.selected_pet_id != selected_pet_id:
                    st.session_state.selected_pet_id = selected_pet_id
                    st.rerun()
                    
        except Exception as e:
            st.dataframe(
                df_display.style.applymap(color_score, subset=['Score']),
                use_container_width=True,
                height=300
            )
            if len(df_display) > 0:
                pet_options = [f"Pet {row['Pet ID']} - Score: {row['Score']} - {row['Primary Concern']}" 
                              for _, row in df_display.iterrows()]
                selected_option = st.radio(
                    "Select a pet:",
                    options=pet_options,
                    index=0
                )
                selected_pet_id = int(selected_option.split(" ")[1])
                if st.session_state.selected_pet_id != selected_pet_id:
                    st.session_state.selected_pet_id = selected_pet_id
                    st.rerun()

    with col2:
        st.header("Pet Details")
        if st.session_state.selected_pet_id:
            pet_rows = sorted_df[sorted_df['pet_id'] == st.session_state.selected_pet_id]
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
