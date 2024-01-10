# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load the movie dataset
url = "https://www.kaggle.com/adrianmcmahon/imdb-india-movies"
movie_data = pd.read_csv(url)

# Data preprocessing
# You may need to customize this based on your dataset
# For example, handling missing values, encoding categorical variables, etc.
# In this example, I'm assuming 'Genre', 'Director', and 'Actors' as features
# and converting them into binary features using one-hot encoding

mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movie_data['Genre'].str.split(',')), columns=mlb.classes_, index=movie_data.index)
directors_encoded = pd.get_dummies(movie_data['Director'], prefix='director')
actors_encoded = pd.DataFrame(mlb.fit_transform(movie_data['Actors'].str.split(',')), columns=mlb.classes_, index=movie_data.index)

# Combine the encoded features with other numerical features
features = pd.concat([genres_encoded, directors_encoded, actors_encoded, movie_data[['Year', 'Runtime', 'Votes']]], axis=1)

# Target variable
target = movie_data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
