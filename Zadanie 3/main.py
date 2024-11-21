"""
Authors: Filip Labuda, Jędrzej Stańczewski
How to use: run the code and provide a name from the given list as an input.
Example output in readme.md
This script loads user ratings from a JSON file, normalizes the ratings, clusters users using K-Means clustering,
and provides movie recommendations based on the user's cluster.
"""

import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load user ratings from the JSON file
with open('./ratings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert JSON data to a pandas DataFrame and transpose it to have users as rows
df = pd.DataFrame(data).transpose()

# Convert all data to numeric, replacing non-numeric entries with NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Normalize ratings for each user by subtracting the user's mean and dividing by the user's standard deviation
normalized_df = (df - df.mean(axis=1).values.reshape(-1, 1)) / df.std(axis=1).values.reshape(-1, 1)

# Replace NaN values with 0 for clustering
normalized_df = normalized_df.fillna(0)

# Apply K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(normalized_df)

# Add cluster labels to the original DataFrame
df['Cluster'] = clusters


def get_recommendations(target_user, df, normalized_df, cluster_column='Cluster', top_n=5):
    """
    Generate movie recommendations and anti-recommendations for a target user.

    Parameters:
        target_user (str): The username for whom to generate recommendations.
        df (pandas.DataFrame): The original DataFrame containing user ratings and cluster labels.
        normalized_df (pandas.DataFrame): The normalized DataFrame used for clustering.
        cluster_column (str): The name of the column containing cluster labels. Default is 'Cluster'.
        top_n (int): The number of top recommendations to return. Default is 5.

    Returns:
        tuple: A tuple containing two lists:
            - recommendations (list): List of top_n recommended movies.
            - anti_recommendations (list): List of top_n least recommended movies.
    """
    # Identify the cluster of the target user
    user_cluster = df.loc[target_user, cluster_column]

    # Get all users in the same cluster
    cluster_users = df[df[cluster_column] == user_cluster].index

    # Calculate average ratings for each movie within the cluster
    cluster_ratings = df.loc[cluster_users].drop(columns=[cluster_column]).mean(axis=0)

    # Get the movies already rated by the target user
    rated_movies = df.loc[target_user].drop(labels=[cluster_column]).dropna().index

    # Exclude movies already rated by the target user from recommendations
    cluster_ratings = cluster_ratings.drop(rated_movies, errors='ignore')

    # Sort movies by average rating to get recommendations and anti-recommendations
    recommendations = cluster_ratings.sort_values(ascending=False).head(top_n)
    anti_recommendations = cluster_ratings.sort_values(ascending=True).head(top_n)

    return recommendations.index.tolist(), anti_recommendations.index.tolist()


# Prompt the user to enter a username from the available users
target_user = input(f"Available users: {', '.join(df.index)}\nEnter the user to get recommendations for: ").strip()

# Check if the entered user exists in the DataFrame
if target_user not in df.index:
    print("Invalid user name. Please try again.")
else:
    recommendations, anti_recommendations = get_recommendations(target_user, df, normalized_df)
    print(f"\nRecommendations for {target_user}: {recommendations}")
    print(f"Anti-Recommendations for {target_user}: {anti_recommendations}")
