#!/usr/bin/env python
# coding: utf-8

# In[7]:

# Importing Libraries to Use
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from tabulate import tabulate
from consolemenu import *
from consolemenu.items import *
import re

#=========================== BEGIN DATA MANIPULATION ==============================#

# Importing Data Files
genre_data = pd.read_csv("movies.csv")
rating_data = pd.read_csv("ratings.csv")

# Creating Mean of Ratings
mean_rating = rating_data.copy()
mean_rating = mean_rating.groupby(['movieId']).mean()


# Merging Genres & Ratings
movie_combined = pd.merge(mean_rating, genre_data, on='movieId')
movie_combined.drop(['timestamp'], axis=1, inplace=True)

# Creating Binary Matrix from Binarizer & Adding Ratings
bin = MultiLabelBinarizer()
matrix = pd.DataFrame(bin.fit_transform(movie_combined['genres'].str.split('|')), columns=bin.classes_, index=movie_combined.index)
matrix['average_rating'] = movie_combined['rating']

#========================== END DATA MANIPULATION ================================#



#============================= BEGIN FUNCTIONS ===================================#

# Function for Returning Recommendations
def get_recommendations_knn(title, recommendations=10):
    
    if title not in movie_combined['title'].values:
        return movie_combined['title'].iloc[0:0]
    
    # KNN Cosine Model for Genre & Rating
    model_knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=15)
    model_knn.fit(matrix.values)
    # Index of Movie
    idx = movie_combined[movie_combined['title'] == title].index[0]

    # Calculate Distance and Indices
    distances, indices = model_knn.kneighbors([matrix.iloc[idx]])

    # Retrieve top User Specified Amount of Recommendations
    genre_movie_indices = indices[0][1:recommendations+1]
    #genre_movie_distances = distances[0][1:recommendations+1]

    # Return Recommendations to Be Printed
    return movie_combined['title'].iloc[list(genre_movie_indices)]

# Function for Main Menu
def display_menu():
    menu = ConsoleMenu("Movie Recommendation Platform")
    function_item = FunctionItem("Search Movie Database", search_movies)
    menu.append_item(function_item)
    function_item = FunctionItem("Find Similar Recommendation", find_recommendation)
    menu.append_item(function_item)
    menu.show()

# Function for Searching Database
def search_movies():
    print("  INPUT SEARCH QUERY >> ")
    movie_name = input()
    movie_search = movie_combined.copy()
    movie_search.drop(['movieId','userId'], axis=1, inplace=True)
    movie_search = movie_search[['title', 'genres', 'rating']]
    movie_search = movie_search[movie_search['title'].str.contains(re.escape(movie_name), case=False)]
    if len(movie_search) > 0:
        table = tabulate(movie_search, headers=["Title", "Genre", "Rating"], tablefmt="fancy_grid", showindex=False, numalign="left", stralign="left", maxcolwidths=35)
        print("\n" + table)
    else:
        print("Movie Not Found in Database")
        

# Function for Returning Recommendations
def find_recommendation():
    print("  INPUT MOVIE >> ")
    movie_name = input()
    recommendations = (get_recommendations_knn(movie_name))
    recommendations = pd.merge(recommendations, movie_combined,on='title')
    recommendations = recommendations[["title","genres","rating"]]
    if len(recommendations) > 0:
        table = tabulate(recommendations, headers=["Title", "Genre", "Rating"], tablefmt="fancy_grid", showindex=False, numalign="left", stralign="left", maxcolwidths=35)
        print("\n" + table) 
    else:
        print("Movie Not Found in Database")

#=============================== END FUNCTIONS ====================================#

# Displaying Menu
display_menu()


# In[ ]:




