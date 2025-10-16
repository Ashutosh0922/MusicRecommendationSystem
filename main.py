import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from streamlit_searchbox import st_searchbox

# ==========================================
#           DATA LOADING FUNCTIONS
# ==========================================

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('ex.csv')
        st.success("✅ Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        st.error("❌ 'ex.csv' file not found! Please ensure it’s in the project directory.")
        return pd.DataFrame()


# ==========================================
#           DATA PREPROCESSING
# ==========================================

def preprocess_data(data):
    if data.empty:
        st.warning("⚠️ No data available for preprocessing.")
        return pd.DataFrame(), None, None

    # Clean and convert User-Rating safely
    if 'User-Rating' in data.columns:
        data['User-Rating'] = data['User-Rating'].astype(str).str.replace('/10', '', regex=False)
        data['User-Rating'] = pd.to_numeric(data['User-Rating'], errors='coerce')
    else:
        st.warning("⚠️ 'User-Rating' column not found in dataset!")

    # Drop rows with missing values
    data_cleaned = data.dropna(subset=['Genre', 'Singer/Artists', 'Album/Movie'])

    # Encode categorical columns
    label_encoder_genre = LabelEncoder()
    label_encoder_singer = LabelEncoder()
    label_encoder_album = LabelEncoder()

    data_cleaned['Genre_Encoded'] = label_encoder_genre.fit_transform(data_cleaned['Genre'])
    data_cleaned['Singer_Encoded'] = label_encoder_singer.fit_transform(data_cleaned['Singer/Artists'])
    data_cleaned['Album_Encoded'] = label_encoder_album.fit_transform(data_cleaned['Album/Movie'])

    return data_cleaned, label_encoder_singer, label_encoder_album

# ==========================================
#           MODEL TRAINING
# ==========================================
@st.cache_data
def train_knn_model(data_cleaned):
    if data_cleaned.empty:
        st.warning("⚠️ Cannot train KNN model — dataset is empty.")
        return None

    num_samples = len(data_cleaned)
    # K must be less than total samples
    n_neighbors = min(6, num_samples)
    
    if n_neighbors <= 1:
        st.warning("⚠️ Not enough data points to train KNN model.")
        return None

    features = data_cleaned[['Genre_Encoded', 'Singer_Encoded', 'Album_Encoded']]
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn_model.fit(features)
    return knn_model


# ==========================================
#           SONG RECOMMENDER
# ==========================================
def recommend_similar_songs(song_name, data, model):
    if model is None or data.empty:
        return []

    try:
        song_index = data[data['Song-Name'] == song_name].index[0]
    except IndexError:
        return []

    num_samples = len(data)
    n_neighbors = min(6, num_samples)

    # If only 1 song, we can’t recommend others
    if n_neighbors <= 1:
        return []

    song_features = data[['Genre_Encoded', 'Singer_Encoded', 'Album_Encoded']].iloc[song_index].values.reshape(1, -1)

    # Find similar songs
    distances, indices = model.kneighbors(song_features, n_neighbors=n_neighbors)
    similar_songs = data.iloc[indices[0]]

    # Drop the song itself if present
    similar_songs = similar_songs[similar_songs['Song-Name'] != song_name]

    return similar_songs[['Song-Name', 'Singer/Artists', 'Genre']].to_dict(orient='records')



# ==========================================
#           LOGGING SYSTEM
# ==========================================

def log_search_result(first_result, log_file='searchedsong.csv'):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    log_data = {
        'Song-Name': first_result['Song-Name'],
        'Singer/Artists': first_result['Singer/Artists'],
        'Genre': first_result['Genre'],
        'Album/Movie': first_result['Album/Movie'],
        'User-Rating': first_result['User-Rating'],
        'search_date': current_date,
        'search_time': current_time
    }

    try:
        existing_log_df = pd.read_csv(log_file)
        updated_log_df = pd.concat([existing_log_df, pd.DataFrame([log_data])], ignore_index=True)
    except FileNotFoundError:
        updated_log_df = pd.DataFrame([log_data])

    updated_log_df.to_csv(log_file, index=False)
    return current_date, current_time


# ==========================================
#           LOG ANALYSIS
# ==========================================

def load_and_preprocess_logged_search_results(log_file='searchedsong.csv'):
    try:
        logged_data = pd.read_csv(log_file)
        return preprocess_data(logged_data)[0]
    except FileNotFoundError:
        return pd.DataFrame(columns=['Song-Name', 'Singer/Artists', 'Genre', 'Album/Movie', 'User-Rating'])


def show_daily_song_preference(data):
    st.subheader("📅 Daily Song Preferences")
    daily_counts = data.groupby('search_date').size().reset_index(name='Count')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='search_date', y='Count', data=daily_counts, palette='Blues_d')
    plt.xticks(rotation=45)
    plt.title("Number of Songs Searched Per Day")
    st.pyplot(plt.gcf())


def show_hourly_song_preference(data):
    st.subheader("⏰ Hourly Song Preferences")
    data['search_hour'] = pd.to_datetime(data['search_time'], format='%H:%M:%S').dt.hour
    hourly_counts = data.groupby('search_hour').size().reset_index(name='Count')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='search_hour', y='Count', data=hourly_counts, palette='Greens_d')
    plt.title("Number of Songs Searched Per Hour")
    st.pyplot(plt.gcf())


def show_average_rating_chart(data):
    st.subheader("⭐ Average User Ratings by Genre")
    avg_ratings = data.groupby('Genre')['User-Rating'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Genre', y='User-Rating', data=avg_ratings, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title("Average Rating per Genre")
    st.pyplot(plt.gcf())


GENRE_MOOD_SCORES = {
    'BollywoodDanceRomantic': 3,
    'BollywoodDance': 2,
    'BollywoodSad': 8,
    'BollywoodRomantic': 5,
    'Bollywood': 4,
    'BollywoodRomance': 6
}

def analyze_today_depression_risk(data_file):
    df = pd.read_csv(data_file)
    current_date = datetime.now().strftime("%Y-%m-%d")
    today_searches = df[df['search_date'] == current_date]
    todays_genre_counts = today_searches['Genre'].value_counts()

    st.write("🎵 Today's Genre Counts:")
    st.write(todays_genre_counts)

    depression_risk_score = 0
    total_genres = 0

    for genre, count in todays_genre_counts.items():
        mood_score = GENRE_MOOD_SCORES.get(genre, 5)
        depression_risk_score += mood_score * count
        total_genres += count

    average_risk_score = depression_risk_score / total_genres if total_genres > 0 else 0
    st.info(f"🧠 **Average Depression Risk Score Today:** {average_risk_score:.2f}")

    if average_risk_score >= 7:
        st.error("⚠️ High risk of depression based on today’s music preferences.")
    elif average_risk_score >= 4:
        st.warning("🟠 Moderate risk of depression based on today’s music preferences.")
    else:
        st.success("🟢 Low risk of depression based on today’s music preferences.")




# ==========================================
#              MAIN APP
# ==========================================
def main():
    st.title("🎧 Mood-Based Music Recommendation System")

    # Load and preprocess dataset
    music_data = load_data()
    if music_data.empty:
        return

    music_data_cleaned, label_encoder_singer, label_encoder_album = preprocess_data(music_data)
    knn_model = train_knn_model(music_data_cleaned)

    st.subheader("🔍 Search for a Song or Artist")

    # Define search function for the auto-suggest bar
    def search_func(term: str):
        if not term:
            return []
        # Return top 10 matching song or artist names
        suggestions = music_data_cleaned[
            music_data_cleaned['Song-Name'].str.contains(term, case=False, na=False) |
            music_data_cleaned['Singer/Artists'].str.contains(term, case=False, na=False)
        ][['Song-Name', 'Singer/Artists']].drop_duplicates().head(10)

        # Combine name + artist for better display
        return [f"{row['Song-Name']} — {row['Singer/Artists']}" for _, row in suggestions.iterrows()]

    # Real-time auto-suggestion input box
    selected_result = st_searchbox(
        search_func,
        key="song_search",
        placeholder="🎵 Type a song or artist name...",
        label="Start typing to get suggestions:"
    )

    # If user selected a suggestion
    if selected_result:
        # Extract song name from selected result
        selected_song = selected_result.split(" — ")[0]
        st.success(f"🎶 You selected: {selected_result}")

        # Fetch search results
        search_results = music_data_cleaned[
            (music_data_cleaned['Song-Name'].str.lower() == selected_song.lower()) |
            (music_data_cleaned['Singer/Artists'].str.lower() == selected_song.lower())
        ][['Song-Name', 'Singer/Artists', 'Genre', 'Album/Movie', 'User-Rating']].drop_duplicates()

        if not search_results.empty:
            st.dataframe(search_results)

            first_result = search_results.iloc[0]
            date, time = log_search_result(first_result)
            st.info(f"🔖 Logged search for '{first_result['Song-Name']}' on {date} at {time}")

            logged_searches = pd.read_csv('searchedsong.csv')
            logged_data_cleaned = load_and_preprocess_logged_search_results()

            # Recommendations from logged data
            if not logged_data_cleaned.empty:
                logged_knn_model = train_knn_model(logged_data_cleaned)
                similar_songs = recommend_similar_songs(first_result['Song-Name'], logged_data_cleaned, logged_knn_model)
                if similar_songs:
                    st.subheader("🎶 Songs similar to your recent searches:")
                    for song in similar_songs:
                        st.write(f"- {song['Song-Name']} by {song['Singer/Artists']} ({song['Genre']})")
            else:
                st.info("No logged searches yet.")

            # Recommendations from main dataset
            similar_songs_main = recommend_similar_songs(first_result['Song-Name'], music_data_cleaned, knn_model)
            if similar_songs_main:
                st.subheader("🎼 Songs similar to your search:")
                for song in similar_songs_main:
                    st.write(f"- {song['Song-Name']} by {song['Singer/Artists']} ({song['Genre']})")

            # Show visual analytics
            show_daily_song_preference(logged_searches)
            show_hourly_song_preference(logged_searches)
            show_average_rating_chart(music_data_cleaned)
            analyze_today_depression_risk('searchedsong.csv')
        else:
            st.warning(f"No matches found for '{selected_song}'. Please try again.")


if __name__ == "__main__":
    main()
