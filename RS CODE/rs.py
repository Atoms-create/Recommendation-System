import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load restaurant data from CSV file
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    st.write("Columns in the dataset:", data.columns)  # Display the column names
    return data

# Calculate cosine similarity between items (restaurants)
def item_based_filtering(data, restaurant_name):
    vectorizer = CountVectorizer()
    item_matrix = vectorizer.fit_transform(data['Restaurant Name'])  # Use restaurant names for similarity
    cosine_sim = cosine_similarity(item_matrix)

    # Find index of the selected restaurant
    idx = data[data['Restaurant Name'] == restaurant_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort restaurants by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Return top 5 similar restaurants
    similar_restaurants = [i[0] for i in sim_scores[1:6]]  # Exclude the selected restaurant itself
    return data.iloc[similar_restaurants]

# Train SVM model to predict restaurant preference based on contextual factors
def train_svm_model(data):
    # Prepare features and labels
    X = data[['Time Of Day', 'Weather', 'Special Occasion']]
    y = data['Restaurant Name']  # Assuming this is the target variable

    # Encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Time Of Day', 'Weather', 'Special Occasion'])
        ]
    )

    # Create and train the SVM model
    model = make_pipeline(preprocessor, SVC(probability=True))
    model.fit(X, y)
    return model

# Train Decision Tree model to predict restaurant preference based on contextual factors
def train_decision_tree_model(data):
    # Prepare features and labels
    X = data[['Time Of Day', 'Weather', 'Special Occasion']]
    y = data['Restaurant Name']

    # Encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Time Of Day', 'Weather', 'Special Occasion'])
        ]
    )

    # Create and train the Decision Tree model
    model = make_pipeline(preprocessor, DecisionTreeClassifier())
    model.fit(X, y)
    return model

# Predict restaurant based on user preferences using SVM
def svm_prediction(model, user_preferences):
    user_data = pd.DataFrame([user_preferences])
    prediction = model.predict(user_data)
    return prediction[0]  # Return the predicted restaurant name

# Predict restaurant based on user preferences using Decision Tree
def decision_tree_prediction(model, user_preferences):
    user_data = pd.DataFrame([user_preferences])
    prediction = model.predict(user_data)
    return prediction[0]  # Return the predicted restaurant name

# Filter restaurants based on contextual factors (time, weather, occasion)
def filter_restaurants(data, time_of_day, weather, occasion):
    filtered_data = data[
        (data['Time Of Day'].str.contains(time_of_day, case=False)) &
        (data['Weather'].str.contains(weather, case=False)) &
        (data['Special Occasion'].str.contains(occasion, case=False))
    ]
    return filtered_data

# Load CSV data
data_file = r"C:\Users\hp\Downloads\chennai_restaurants2.csv"  # Use raw string to handle backslashes
data = load_data(data_file)

# Train the SVM and Decision Tree models
svm_model = train_svm_model(data)
decision_tree_model = train_decision_tree_model(data)

# Streamlit chatbot UI
st.title('Personalized Dining Suggestion Chatbot')

# Get user inputs
st.subheader('Provide your preferences:')
time_of_day = st.text_input('Enter the time of day (e.g., "Morning", "Afternoon", "Night"):', value='Night')
weather = st.text_input('Enter the current weather (e.g., "Rainy", "Sunny"):', value='Sunny')
occasion = st.text_input('Enter any special occasion (e.g., "Birthday", "Anniversary"):', value='Birthday')
restaurant_choice = st.text_input('Enter a restaurant you like (for item-based recommendations):', value='')

# Button to generate suggestions
if st.button('Get Dining Suggestions'):
    try:
        filtered_restaurants = filter_restaurants(data, time_of_day, weather, occasion)
        
        if not filtered_restaurants.empty:
            st.subheader('Here are some personalized dining suggestions based on your context:')
            for index, row in filtered_restaurants.iterrows():
                st.write(f"Name: {row['Restaurant Name']}")
                st.write(f"Address: {row['Address']}")
                st.write(f"Rating: {row['Rating']}")
                st.write('---')
        else:
            st.write('.')
    except KeyError as e:
        st.write(f"Error: Column {e} not found in the dataset. Please check the column names in the CSV file.")

    # Item-based filtering
    if restaurant_choice:
        st.subheader(f'Restaurants similar to {restaurant_choice}:')
        try:
            similar_restaurants = item_based_filtering(data, restaurant_choice)
            for index, row in similar_restaurants.iterrows():
                st.write(f"Name: {row['Restaurant Name']}")
                st.write(f"Address: {row['Address']}")
                st.write(f"Rating: {row['Rating']}")
                st.write('---')
        except IndexError:
            st.write('.')

    # User-based filtering using SVM
    st.subheader('Recommendations based on user preferences (SVM):')
    user_preferences = {'Time Of Day': time_of_day, 'Weather': weather, 'Special Occasion': occasion}
    predicted_restaurant_svm = svm_prediction(svm_model, user_preferences)
    
    st.write(f"Recommended Restaurant (SVM): {predicted_restaurant_svm}")

    # User-based filtering using Decision Tree
    st.subheader('Recommendations based on user preferences (Decision Tree):')
    predicted_restaurant_dt = decision_tree_prediction(decision_tree_model, user_preferences)
    
    st.write(f"Recommended Restaurant (Decision Tree): {predicted_restaurant_dt}")
