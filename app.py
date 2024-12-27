import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
MODEL_FEATURES = ['racePosition', 'weighted_wins', 'avg_position', 'podiums', 'normalized_points']
# Load and preprocess the data
@st.cache_data
def load_data():
    races_data = pd.read_csv('Last_Races_of_Each_Formula_1_Season.csv')
    merged_data = pd.read_csv('Cleaned_Merged_Formula_1_Dataset.csv')
    return races_data, merged_data

def filter_final_race_data(races_data, merged_data):
    # Filter the merged dataset to include only the last races of each season
    final_race_ids = races_data['raceId'].unique()
    final_race_data = merged_data[merged_data['raceId'].isin(final_race_ids)]
    return final_race_data

def preprocess_data(data):
    # Convert 'racePosition' to numeric, treating non-numeric values as NaN
    data['racePosition'] = pd.to_numeric(data['racePosition'], errors='coerce')
    
    # Find the maximum numeric position
    max_position = data['racePosition'].max()
    
    # Replace NaN (which includes 'R' and other non-numeric values) with max_position + 1
    data['racePosition'] = data['racePosition'].fillna(max_position + 1)
    
    # Convert to integer type
    data['racePosition'] = data['racePosition'].astype(int)
    
    # Calculate 'wins' for each driver in each season
    data['wins'] = data.groupby(['year', 'driverId'])['racePosition'].transform(lambda x: (x == 1).sum())
    
    # Create a weighted wins feature
    data['weighted_wins'] = data['wins'] * 5  # Multiply wins by 5 to give it more weight
    
    # Calculate average finishing position for each driver in each season
    data['avg_position'] = data.groupby(['year', 'driverId'])['racePosition'].transform('mean')
    
    # Calculate podium finishes (top 3 positions)
    data['podiums'] = data.groupby(['year', 'driverId'])['racePosition'].transform(lambda x: (x <= 3).sum())
    
    # Normalize points within each season
    data['normalized_points'] = data.groupby('year')['points'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Determine the champion for each season
    data['champion'] = data.groupby('year')['points'].transform(max) == data['points']
    data['champion'] = data['champion'].astype(int)
    
    return data
def train_model(data):
    X = data[MODEL_FEATURES]
    y = data['champion']

    # Handle class imbalance using resampling
    X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=42)
    X_resampled = pd.concat([X[y == 0], X_resampled])
    y_resampled = pd.concat([y[y == 0], y_resampled])

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return model, X_test, y_test, y_pred, accuracy, roc_auc
def predict_championship(model):
    race_position = st.sidebar.number_input('Race Position', min_value=1, max_value=100, step=1)
    weighted_wins = st.sidebar.number_input('Wins', min_value=0, max_value=100, step=1)
    avg_position = st.sidebar.number_input('Average Position', min_value=1, max_value=100, step=1)
    podiums = st.sidebar.number_input('Podiums', min_value=0, max_value=100, step=1)
    normalized_points = st.sidebar.number_input('Points', min_value=0, max_value=1000, step=1)

    input_data = pd.DataFrame({
        'racePosition': [race_position],
        'weighted_wins': [weighted_wins],
        'avg_position': [avg_position],
        'podiums': [podiums],
        'normalized_points': [normalized_points]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.sidebar.subheader("Championship Prediction")
    if prediction[0] == 1:
        st.sidebar.success("This driver is predicted to be the champion!")
    else:
        st.sidebar.error("This driver is not predicted to be the champion.")
    st.sidebar.write(f"Probability of being champion: {probability:.2%}")


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(feature_names, importance)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

def plot_driver_performance(data):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['qualiPosition'], data['racePosition'], c=data['points'], cmap='viridis')
    ax.set_xlabel('Qualification Position')
    ax.set_ylabel('Race Position')
    ax.set_title('Driver Performance vs Qualification')
    fig.colorbar(scatter, ax=ax, label='Points')
    st.pyplot(fig)


def main():
    st.title("Formula 1 World Championship Prediction")
    
    races_data, merged_data = load_data()
    final_race_data = filter_final_race_data(races_data, merged_data)
    processed_data = preprocess_data(final_race_data)

    # Add 'wins' column if it doesn't exist
    if 'wins' not in processed_data.columns:
        processed_data['wins'] = processed_data.groupby('driverId')['racePosition'].apply(lambda x: (x == 1).sum())

    model, X_test, y_test, y_pred, accuracy, roc_auc = train_model(processed_data)


    # Call the new prediction function
    predict_championship(model)

    st.write("### Final Races Data")
    display_data = final_race_data[MODEL_FEATURES + ['champion']].head().rename(
        columns={
            'racePosition': 'Race Position',
            'weighted_wins': 'Wins',
            'avg_position': 'Average Position',
            'podiums': 'Podiums',
            'normalized_points': 'Points',
            'champion': 'Champion'
        }
    )
    st.write(display_data)

    st.write("### Data Insights")
    plot_driver_performance(processed_data)



    st.write("### Feature Importance")
    plot_feature_importance(model, X_test.columns)

if __name__ == "__main__":
    main()