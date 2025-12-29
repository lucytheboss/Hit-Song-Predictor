import streamlit as st
import pandas as pd
import pickle
import numpy as np

# SETUP & LOAD MODEL
st.set_page_config(page_title="Hit Song Simulator", page_icon="🎵")

@st.cache_resource
def load_model():
    with open('spotify_model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    return artifact['model'], artifact['columns']

try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Error: 'spotify_model.pkl' not found. Please run Part 1 in your notebook first!")
    st.stop()

# 2. SIDEBAR (USER INPUTS)
st.sidebar.header("🎛️ Song Studio")
st.sidebar.write("Tweak the variables to see the predicted success.")

fame = st.sidebar.slider("Artist Fame", 0, 100, 50, help="Long-term brand popularity")
momentum = st.sidebar.slider("Momentum (Prev Hit)", 0, 100, 50, help="Popularity of the last released track")

duration_min = st.sidebar.number_input("Duration (Minutes)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)

genre_list = ['Christian', 'Christmas: Pop', 'Classical', 'Country', 'Dance',
            'Easy Listening', 'Electronic', 'Hard Rock', 'Hip-Hop', 'Hip-Hop/Rap',
            'Holiday', 'Metal', 'Música tropical', 'Pop', 'R&B/Soul', 'Rock',
            'Singer/Songwriter', 'Soft Rock', 'Soundtrack']

genre_cols = [c for c in model_columns if c in genre_list]
clean_genres = [c.replace('Genre_', '') for c in genre_cols]
clean_genres.append('Other')
selected_genre = st.sidebar.selectbox("Primary Genre", sorted(clean_genres))


# 3. PREDICTION ENGINE
def make_prediction():
    input_data = {col: 0 for col in model_columns}

    input_data['const'] = 1.0 
    input_data['artistPopularity'] = fame
    input_data['prevTrackPopularity'] = momentum
    
    genre_key = f"Genre_{selected_genre}"
    if genre_key in input_data:
        input_data[genre_key] = 1
        
    if duration_min < 2.0:
        bin_label = 'Very Short' 
    elif 2.0 <= duration_min < 3.0:
        if 'Duration_Short' in input_data: input_data['Duration_Short'] = 1
    elif 3.0 <= duration_min < 4.5:
        if 'Duration_Standard' in input_data: input_data['Duration_Standard'] = 1
    elif 4.5 <= duration_min < 6.0:
        if 'Duration_Long' in input_data: input_data['Duration_Long'] = 1
    else: # > 6.0
        if 'Duration_Very Long' in input_data: input_data['Duration_Very Long'] = 1

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return max(0, min(100, prediction))

predicted_score = make_prediction()

# 4. MAIN DASHBOARD UI
st.title("🎵 Hit Song Predictor")
st.markdown("### Based on Data from 2,000+ Tracks")

# Top Section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Predicted Popularity", value=f"{predicted_score:.1f}/100")

with col2:
    if predicted_score > 80:
        st.write("## 🔥 Smash Hit")
    elif predicted_score > 60:
        st.write("## ✅ Solid Performance")
    elif predicted_score > 40:
        st.write("## ⚠️ Risk of Flop")
    else:
        st.write("## 🧊 Cold Release")

# Middle Section
if duration_min < 2.0:
    st.warning("⚠️ **Duration Warning:** Your song is under 2 minutes. The model predicts a ~20 point penalty for tracks that are too short.")
elif 3.0 <= duration_min <= 4.5:
    st.success("✅ **Sweet Spot:** You are in the ideal duration range (3:00 - 4:30) for maximum popularity.")

# Bottom Section
st.markdown("---")
st.subheader("💡 Why this score?")
st.write(f"""
* **Base Fame:** Your artist fame contributes significantly (baseline).
* **Momentum:** The previous track's success ({momentum}) is pulling the score {'up' if momentum > 60 else 'down'}.
* **Genre:** Selected genre is **{selected_genre}**.
* **Duration:** {duration_min} minutes.
""")

# Debug option
with st.expander("See Raw Model Input"):
    st.write("This is exactly what the model sees (1 = Active, 0 = Inactive):")
    dummy_data = {col: 0 for col in model_columns}
    st.write(f"Model Columns: {model_columns[:5]} ...")
    
# 5. BATCH PREDICTION (CSV UPLOAD)
st.markdown("---")
st.header("📂 Batch Analysis")
st.write("Upload a CSV of new songs to predict them all at once.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load the Data
    new_data = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(new_data)} songs.")
    
    # Process Data (Replicating the Logic)
    processed_df = new_data.copy()
    
    # Calculate duration in minutes if needed
    if 'trackTimeMillis' in processed_df.columns and 'durationMins' not in processed_df.columns:
        processed_df['durationMins'] = processed_df['trackTimeMillis'] / 60000
    # Binning
    bins = [0, 2.0, 3.0, 4.5, 6.0, float('inf')]
    labels = ['Very Short', 'Short', 'Standard', 'Long', 'Very Long']
    processed_df['duration_granular'] = pd.cut(processed_df['durationMins'], bins=bins, labels=labels)
    
    # Create Dummies
    genre_dummies = pd.get_dummies(processed_df['primaryGenreName'], prefix='Genre', drop_first=True)
    dur_dummies = pd.get_dummies(processed_df['duration_granular'], prefix='Duration', drop_first=True)
    
    processed_df = pd.concat([processed_df, genre_dummies, dur_dummies], axis=1)
    
    # 3. ALIGNMENT (Crucial Step)
    final_input = pd.DataFrame(0, index=processed_df.index, columns=model_columns)
    
    # Fill in the matches
    common_cols = processed_df.columns.intersection(model_columns)
    final_input[common_cols] = processed_df[common_cols]
    
    # Ensure constants
    if 'const' in model_columns:
        final_input['const'] = 1.0
        
    # 4. Predict
    predictions = model.predict(final_input)
    
    # 5. Show Results
    new_data['Predicted_Score'] = predictions
    st.dataframe(new_data[['trackName', 'artistName', 'Predicted_Score']].sort_values(by='Predicted_Score', ascending=False))
    
    # Download Button
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions",
        csv,
        "predicted_hits.csv",
        "text/csv",
        key='download-csv'
    )