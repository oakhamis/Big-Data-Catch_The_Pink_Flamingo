"""
Exploratory Data Analysis
"""

# Importing necessary libraries
import pandas as pd
import seaborn as sns
import os

# Getting and setting the working directory
os.getcwd()
os.chdir('C:/Users\Omar\Desktop\Spring 2020\Big Data Management\Assessment\Report')

############################## DATA PREPROCESSING ############################

# Loading user information and performing initial data preprocessing
user_info = pd.read_csv ('users.csv')
user_info = user_info.rename(columns=lambda x: x.strip())  # Removing whitespaces from column headers
user_info['age'] = pd.to_datetime('today').year - pd.to_datetime(user_info['dob']).dt.year  # Calculating age from date of birth
del user_info['dob']  # Deleting date of birth column

# Loading session information and performing data preprocessing
session_info = pd.read_csv ("user-session.csv")
session_info = session_info.rename(columns=lambda x: x.strip())
session_info = session_info[['userId','teamId','platformType']]
session_info = session_info.drop_duplicates(['userId']).reset_index(drop=True)

# Merging user information and session information
combined_info = pd.merge(user_info, session_info, how='outer', suffixes=('', '_drop'))

# Loading team information and performing data preprocessing
team_info = pd.read_csv ("team.csv")
team_info = team_info.rename(columns=lambda x: x.strip())
team_info = team_info[['teamId','strength']]
combined_info = pd.merge(combined_info, team_info, how='outer', suffixes=('', '_drop'))

# Further processing to get team user count
team_user = combined_info.copy()
team_user = combined_info[['teamId','userId']]
team_user = team_user[team_user['teamId'].notna()]
team_user_count = team_user.groupby('teamId')['userId'].count().to_frame().reset_index()
team_user_count = team_user_count.rename(columns = {'userId':'users_count'})

# Loading buy information and performing data preprocessing
buy_info = pd.read_csv ('buy-clicks.csv')
buy_info = buy_info.rename(columns=lambda x: x.strip())
buy_count = buy_info.groupby(['userId'], as_index=False).agg({'price':'sum', 'buyId':'count'})
combined_info = pd.merge(combined_info, buy_count, how='outer', suffixes=('', '_drop'))

# Loading game information and performing data preprocessing
game_info = pd.read_csv ('game-clicks.csv')
game_info = game_info.rename(columns=lambda x: x.strip())
game_count = game_info.groupby(['userId'], as_index=False).agg({'isHit':'sum', 'clickId':'count'})
combined_info = pd.merge(combined_info, game_count, how='outer', suffixes=('', '_drop'))

# Loading ad information and performing data preprocessing
ad_info = pd.read_csv ('ad-clicks.csv')
ad_info = ad_info.rename(columns=lambda x: x.strip())
ad_count = ad_info.groupby(['userId'], as_index=False).agg({'adId':'count'})
combined_info = pd.merge(combined_info, ad_count, how='outer', suffixes=('', '_drop'))

# Dropping duplicate columns
combined_info.drop([col for col in combined_info.columns if 'drop' in col], axis=1, inplace=True)

# Finalizing the combined dataframe
combined_final = combined_info[combined_info['userId'].notna()]
combined_final = combined_final.rename(columns={
    'price': 'price_sum',
    'clickId': 'game_clicks_count',
    'isHit': 'hit_count' ,
    'buyId': 'buy_clicks_count',
    'adId': 'ad_clicks_count'
})
combined_final['hit_rate'] = (combined_final['hit_count'] / combined_final['game_clicks_count'])
combined_final['ranking_class'] = np.where((combined_final['hit_count'] / combined_final['game_clicks_count']) > 0.110162, 1, 0)
combined_final['total_clicks'] = (combined_final['buy_clicks_count'] + combined_final['ad_clicks_count'] + combined_final['game_clicks_count'])

################################# VIZUALIZATIONS #############################

# Various visualizations to understand the data distribution
plt.figure(figsize=(9,9))
combined_final.groupby('platformType').size().plot(kind='pie', autopct='%.1f')
plt.axis('equal')
plt.legend(loc=2,fontsize=10)
plt.ylabel('')
plt.show()

sns.catplot(x="buy_clicks_count",y="platformType",kind='box',data=combined_final, showfliers=False)
sns.catplot(x="game_clicks_count",y="platformType",kind='box',data=combined_final, showfliers=False)
sns.catplot(x="ad_clicks_count",y="platformType",kind='box',data=combined_final, showfliers=False)
sns.catplot(x="hit_count",y="platformType",kind='box',data=combined_final, showfliers=False)

# Descriptive statistics and grouping by platform type
Flamingo_summary = combined_final.describe()
Flamingo_summary = Flamingo_summary.T
Flamingo_summary.drop(['userId', 'teamId', 'ranking_class'], axis=0, inplace=True)
Flamingo_plat_group = combined_final.groupby('platformType').mean()
del Flamingo_plat_group['userId']
del Flamingo_plat_group['teamId']
del Flamingo_plat_group['ranking_class']

# Heatmap for correlation analysis
plt.figure(figsize=(12,10), dpi=80)
sns.heatmap(combined_final.corr(), xticklabels=combined_final.corr().columns,
            yticklabels=combined_final.corr().columns, center=0,
            annot=True, square=False, cmap='coolwarm', linewidths=3, linecolor='black',
            vmin=-1, vmax=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
corr = combined_final.corr()

################################## WORLD MAP PLOT ############################

# Getting users' countries and plotting them on a world map
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="http")
coordinates = {'latitude': [], 'longitude': []}
for count, user_loc in enumerate(combined_final['country']):
    try:
        location = geolocator.geocode(user_loc)
        if location:
            coordinates['latitude'].append(location.latitude)
            coordinates['longitude'].append(location.longitude)
    except:
        pass
    
Countrydf = pd.DataFrame.from_dict(coordinates,orient='index').transpose()

# Creating a world map to show distributions of users
import folium
from folium.plugins import MarkerCluster
import webbrowser
world_map= folium.Map(tiles="cartodbpositron", zoom_start=1.5)
marker_cluster = MarkerCluster().add_to(world_map)
for i in range(len(Countrydf)):
    lat = Countrydf.iloc[i]['latitude']
    long = Countrydf.iloc[i]['longitude']
    radius=5
    popup_text = """Country : {}<br>
                %of Users : {}<br>"""
    popup_text = popup_text.format(Countrydf.iloc[i]['latitude'],
                               Countrydf.iloc[i]['longitude']
                               )
    folium.CircleMarker(location=[lat, long], radius=radius, popup=popup_text, fill=True).add_to(marker_cluster)
world_map.save("map.html")
webbrowser.open("map.html")

############################ Classification ##################################

# Preparing the dataset for classification
class_hit = combined_final.copy()
class_hit.isna().any()
del class_hit['timestamp']
del class_hit['nick']
del class_hit['twitter']
class_hit['platformType'] = class_hit['platformType'].fillna('No Platform')
class_hit['country'] = class_hit['country'].fillna('No Country')
class_hit['teamId'] = class_hit['teamId'].fillna(0)
class_hit['strength'] = class_hit['strength'].fillna(0)
class_hit['price_sum'] = class_hit['price_sum'].fillna(0)
class_hit['buy_clicks_count'] = class_hit['buy_clicks_count'].fillna(0)
class_hit['hit_count'] = class_hit['hit_count'].fillna(0)
class_hit['game_clicks_count'] = class_hit['game_clicks_count'].fillna(0)
class_hit['ad_clicks_count'] = class_hit['ad_clicks_count'].fillna(0)
class_hit['hit_rate'] = class_hit['hit_rate'].fillna(0)
class_hit.isna().any()

# Encoding categorical variables for classification
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
PT_label_encoder = enc.fit(class_hit['platformType'])
PT_integer_classes = PT_label_encoder.transform(PT_label_encoder.classes_)
t = PT_label_encoder.transform(class_hit['platformType'])
class_hit['EncPlatformType'] = t
CO_label_encoder = enc.fit(class_hit['country'])
CO_integer_classes = CO_label_encoder.transform(CO_label_encoder.classes_)
t = CO_label_encoder.transform(class_hit['country'])
class_hit['EncCountry'] = t

# Further data preprocessing for classification
class_hit = class_hit.drop(class_hit[(class_hit['teamId'] == 0) 
                                     & (class_hit['strength'] == 0) 
                                     & (class_hit['game_clicks_count'] == 0) 
                                     & (class_hit['buy_clicks_count'] == 0)].index)

# Visualizing class variable distribution before dataset balancing
Gavg = 0
Lavg = 0
for index, row in combined_final.iterrows():    
    if row["ranking_class"] > 0:
       Gavg = Gavg + 1    
    else:
       Lavg = Lavg + 1

labels = 'Hit Rate more than average', 'Hit Rate less than average'
sizes = [Gavg, Lavg]
colors = ['yellowgreen','lightcoral']
explode = (0, 0) 
plt.title('PINK FLAMINGO USER HIT RATE') 
plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
plt.axis('equal')  
plt.show()

# Removing outliers for classification
for x in ['hit_rate']:
    q75, q25 = np.percentile(class_hit.loc[:,x],[75,25])
    intr_qr = q75 - q25
    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)
    class_hit.loc[class_hit[x] < min, x] = np.nan
    class_hit.loc[class_hit[x] > max, x] = np.nan

class_hit = class_hit.dropna(axis=0)
class_hit.isnull().sum()
Class_hit_final = class_hit.copy()
del Class_hit_final['platformType']
del Class_hit_final['country']
del Class_hit_final['userId']
Class_hit_final.isna().any()

# Visualizing class variable distribution after dataset balancing and other visualizations
sns.catplot(x="hit_rate",kind='box',data=combined_final)
sns.catplot(x="hit_rate",kind='box',data=Class_hit_final)
Flamingo_summary_Final = Class_hit_final.describe()
Flamingo_summary_Final = Flamingo_summary_Final.T
Flamingo_summary_Final
Gavg = 0
Lavg = 0
for index, row in Class_hit_final.iterrows():    
    if row["ranking_class"] > 0:
       Gavg = Gavg + 1    
    else:
       Lavg = Lavg + 1

labels = 'Hit Rate more than average', 'Hit Rate less than average'
sizes = [Gavg, Lavg]
colors = ['yellowgreen','lightcoral']
explode = (0, 0) 
plt.title('PINK FLAMINGO USER HIT RATE') 
plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
plt.axis('equal')  
plt.show()

################# Classification Accuracy & Confusion Matrix #################

# Splitting the data into training and testing sets for classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
X = Class_hit_final.drop('ranking_class', axis=1)
Y = Class_hit_final['ranking_class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

# Training a Naive Bayes classifier and evaluating its performance
clf = GaussianNB()
clf = clf.fit(x_train, y_train)
Y_prediction = clf.predict(x_test)
print("Accuracy for Naive Bayes: ", accuracy_score(y_test, Y_prediction))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
Y_prediction = clf.predict(x_test)
print(confusion_matrix(y_test, Y_prediction))
report = classification_report(y_test, Y_prediction)
Y_prediction  = clf.predict(x_test)
Class_hit_evaluation = pd.concat([x_test, y_test], axis=1)
Class_hit_evaluation['Predicted_rate'] = Y_prediction
Class_hit_evaluation[['Predicted_rate', 'ranking_class']]
y_true = Class_hit_evaluation['ranking_class']
y_pred = Class_hit_evaluation['Predicted_rate']
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
import numpy as np
unique_label = np.unique(y_true)
print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=unique_label), 
                   index=['actual:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))
cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=unique_label), 
                   index=['actual:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label])
sns.heatmap(cm, annot=True,
         fmt='.2f',
         xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.ylabel('Actual hit rate')
plt.xlabel('Predicted hit rate')

################################# CLUSTERING #################################

# Initializing Spark and preparing data for clustering
import findspark
import pyspark
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
findspark.init('C:\spark')
ssb = SparkSession.builder \
           .master('local[*]') \
           .appName('PF app') \
           .getOrCreate()
print(pyspark.__version__)
Clust_DF = Class_hit_final.copy()
Clust_DF.isna().any()

# Selecting relevant features for K-means clustering
Kmeans_headers = Clust_DF[['buy_clicks_count','ad_clicks_count']]
pDF = ssb.createDataFrame(Kmeans_headers)
parsedData = pDF.rdd.map(lambda line: array([line[0],line[1]]))

# Determining the optimal number of clusters using the Elbow Method
from math import sqrt
def error(point): 
    center = clusters.centers[clusters.predict(point)] 
    return sqrt(sum([x**2 for x in (point - center)]))
kval = []
wssse = []
for k in range(2,10):
    clusters = KMeans.train(parsedData, k=k, maxIterations=10,runs=10, initializationMode="random")
    wssse.append(parsedData.map(lambda point:error(point)).reduce(lambda x, y: x + y))
    kval.append(k)
from pandas import DataFrame
kval_df = DataFrame(kval,columns=['k'])
wssse_df = DataFrame(wssse,columns=['wssse'])
K_wssse_final= pd.concat([kval_df,wssse_df], axis=1)
plt.plot(kval_df, wssse_df)
plt.xlabel("K Value")
plt.ylabel("WSSSE")
plt.show()

# Training the K-means model with the optimal number of clusters (k=3)
clusters_Final = KMeans.train(parsedData, 3, maxIterations=10, initializationMode="random")
WSSSE_Final = (parsedData.map(lambda point:error(point)).reduce(lambda x, y: x + y))
print('k= ' + str(3) + ' WSSSE = ' + str(WSSSE_Final))
print(clusters_Final.centers)
Cluster_centers = clusters_Final.centers
ssb.stop

################################# GRAPH ANALYSIS #############################

# Preparing data for graph analysis
Most_active_teams = combined_final.copy()
Most_active_teams = Most_active_teams[['teamId','total_clicks']]
Most_active_teams = Most_active_teams[Most_active_teams['teamId'].notna()]
Most_active_teams = Most_active_teams.groupby('teamId')['total_clicks'].sum().to_frame().reset_index()
Most_active_teams = Most_active_teams[Most_active_teams.teamId != 0]
Most_active_teams.sort_values(by='total_clicks', ascending=False, inplace=True)
Most_active_teams = Most_active_teams.reset_index(drop=True)

##############################################################################

