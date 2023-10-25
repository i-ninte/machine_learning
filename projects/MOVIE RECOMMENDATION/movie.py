# Import necessary libraries
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination

# Load movie ratings data (You need to have a dataset with user ratings)
data = pd.read_csv('movie_ratings.csv')

# Define the Bayesian Network structure
model = BayesianNetwork([('Genre', 'Rating'), ('Actor', 'Rating'), ('Director', 'Rating')])

# Estimate CPDs (Conditional Probability Distributions) from data
pe = ParameterEstimator(model, data)
model.fit(data, estimator=pe)

# Perform inference using Variable Elimination
inference = VariableElimination(model)

# Function to recommend movies for a user
def recommend_movies(user_input):
    query = inference.map_query(variables=['Movie'], evidence=user_input)
    recommended_movies = query['Movie']
    return recommended_movies

# User input (You can customize this part based on user preferences)
user_preferences = {'Genre': 'Action', 'Actor': 'Tom Hanks', 'Director': 'Steven Spielberg'}

# Get movie recommendations for the user
recommended_movies = recommend_movies(user_preferences)

# Print recommended movies
print("Recommended Movies:")
for movie in recommended_movies:
    print(movie)
