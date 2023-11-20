# Bayesian Network Movie Recommender README

This project implements a Movie Recommender System using Bayesian Network modeling. The system recommends movies based on user preferences for genres, actors, and directors. The Bayesian Network is built using the pgmpy library, and inference is performed using Variable Elimination.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Input](#input)
4. [Output](#output)
5. [Example](#example)
6. [Acknowledgments](#acknowledgments)

## 1. Installation

Ensure you have the required libraries installed. You can install them using the following:

```bash
pip install pandas pgmpy
```
## USAGE
Import necessary libraries:

python
Copy code
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
Load movie ratings data. Ensure you have a dataset with user ratings (e.g., 'movie_ratings.csv').

Define the Bayesian Network structure:



model = BayesianNetwork([('Genre', 'Rating'), ('Actor', 'Rating'), ('Director', 'Rating')])
Estimate CPDs (Conditional Probability Distributions) from data:


pe = ParameterEstimator(model, data)
model.fit(data, estimator=pe)
Perform inference using Variable Elimination:

python
Copy code
inference = VariableElimination(model)
Create a function to recommend movies for a user:

python
Copy code
def recommend_movies(user_input):
    query = inference.map_query(variables=['Movie'], evidence=user_input)
    recommended_movies = query['Movie']
    return recommended_movies


3. Input
The input is a dictionary representing the user's preferences:

python
Copy code
user_preferences = {'Genre': 'Action', 'Actor': 'Tom Hanks', 'Director': 'Steven Spielberg'}
4. Output
The output is a list of recommended movies.

5. Example
See the provided script for an example of how to use the Movie Recommender System.

python
Copy code
# ... (same preamble)

# User input (You can customize this part based on user preferences)
user_preferences = {'Genre': 'Action', 'Actor': 'Tom Hanks', 'Director': 'Steven Spielberg'}

# Get movie recommendations for the user
recommended_movies = recommend_movies(user_preferences)

# Print recommended movies
print("Recommended Movies:")
for movie in recommended_movies:
    print(movie)
6. Acknowledgments
The project utilizes the pgmpy library for Bayesian Network modeling.
The dataset used for movie ratings is not provided in this repository. Ensure you have your own dataset with user ratings.
Feel free to adapt and integrate this Movie Recommender System into your projects!
