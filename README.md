# Catch the Pink Flamingo:

## Case study on user engagement in relation to in-game interactions

# Exploratory Data Analysis on Pink Flamingo Gaming Data

This repository houses the Exploratory Data Analysis (EDA) on Pink Flamingo's gaming data. We dive deep into understanding user behaviors, hit rates, platform usages, and more. The analysis pipeline is divided into multiple sections: data preprocessing, visualizations, geospatial plotting, classification, clustering, and graph analysis.

## Data Preprocessing

This section focuses on importing and cleaning multiple datasets. We have:
- User information
- Session data
- Team information
- Buy-click data
- Game-click data
- Ad-click data

The goal is to merge these datasets into a single, clean dataframe for further analysis.

## Visualizations

Several visualizations are used to understand the distribution of user behaviors across different platforms and their hit rates in the game.

## Geospatial Plotting

We map out the user distributions across different countries, giving an idea about where the game is popular.

## Classification

The goal here is to classify users based on their hit rates. We process the data, balance the datasets, remove outliers, and visualize the class distributions. A Naive Bayes classifier is trained to predict the user's ranking class based on their behaviors.

## Clustering

Using the K-means algorithm, we cluster users based on their buy and ad click behaviors. The optimal number of clusters is determined using the Elbow Method.

## Graph Analysis

The graph analysis section is focused on understanding the most active teams based on total clicks.

## Dependencies
Before running the script, ensure you have the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `geopy`
- `folium`
- `sklearn`
- `findspark`
- `pyspark`

## Usage
1. Set the working directory to the location of your datasets.
2. Ensure you have all necessary dependencies installed.
3. Run the script to perform the EDA.

## Contributing
If you'd like to contribute, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.

## License
Distributed under the MIT License. See `LICENSE` for more information.
