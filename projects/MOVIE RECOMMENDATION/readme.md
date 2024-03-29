(Link to Netflix Dashboard on Tableau)[https://public.tableau.com/app/profile/kwabena.obeng/viz/NextflixDashboard/Netflix]


# Bayesian Network Movie Recommender 

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



inference = VariableElimination(model)
Create a function to recommend movies for a user:


def recommend_movies(user_input):
    query = inference.map_query(variables=['Movie'], evidence=user_input)
    recommended_movies = query['Movie']
    return recommended_movies


## Input
The input is a dictionary representing the user's preferences:


user_preferences = {'Genre': 'Action', 'Actor': 'Tom Hanks', 'Director': 'Steven Spielberg'}
## Output
The output is a list of recommended movies.

## Example
See the provided script for an example of how to use the Movie Recommender System.




# User input (You can customize this part based on user preferences)
user_preferences = {'Genre': 'Action', 'Actor': 'Tom Hanks', 'Director': 'Steven Spielberg'}

# Get movie recommendations for the user
recommended_movies = recommend_movies(user_preferences)

# Print recommended movies
print("Recommended Movies:")
for movie in recommended_movies:
    print(movie)

    
## Acknowledgments
The project utilizes the pgmpy library for Bayesian Network modeling.
(link to dataset)[https://github.com/DataScienceRoadMapDSRM/Tableau-Dashboards-info/blob/main/netflix_titles.csv]
Feel free to adapt and integrate this Movie Recommender System into your projects!
