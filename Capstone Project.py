#!/usr/bin/env python
# coding: utf-8

# # CAPSTONE PROJECT - THE BATTLE OF NEIGHBORHOODS

# # What is the best place to open a Restaurant in Paris, France?
# 

# # 1. Introduction

# I have been constantly striving to develop my technical skills to become a Data Science. So, I took the IBM course in order to gain knowledge and pursue the IBM Data Science Professional Certification: https://www.coursera.org/professional-certificates/ibm-data-science. 
# 
# During this course, I learned how to use Data Science tools, such as Jupyter Notebook, GitHub and IBM Watson Studio. The main programming language used was Python, which is packed with powerful libraries that can be utilised for Data Science such as Pandas, Numpy, Matplotlib, Seaborn, Folium, Scikit-learn and SciPy.
# 
# In the final assignment, called “Capstone Project”, it was required to use various tools and methodologies learned throughout this course to solve a real-life business problem. This business problem had to involve the use of location data derived from Foursquare (https://foursquare.com) using API.
# 
# 

# # 2. Business Problem

# Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). Since the 17th century, Paris has been one of Europe's major centres of finance, diplomacy, commerce, fashion, gastronomy, science, and arts.
# 
# The city of Paris is divided into twenty arrondissements municipaux, administrative districts, more simply referred to as arrondissements. These are not to be confused with departmental arrondissements, which subdivide the larger French départements. The word "arrondissement", when applied to Paris, refers almost always to the municipal arrondissements presented in the figure below.

# ![image-2.png](attachment:image-2.png)

# 
# There are currently around 12,000 restaurants in Paris for 2,150 million inhabitants. That is to say, on average, about 1 restaurant per hectare. That's a lot, the competition is tough. Many restaurants are closing, especially with the coronavirus crisis. For who are think about opening a new restaurant, a market research that indicates the best commercial spot is very important and can be a big step towards business success.
# 

# ## 2.1 Objective and interest

# ## 2.1. Business Understanding
# 

# The aim of this project is to find the best neighborhood of Paris to open a new restaurant. For it, it will be showed the global vision of the distribution of restaurants in Paris.
# 

# ## 2.2. Analytical Approach

# The total number of neighborhoods in Paris are 20, so it is necessary to find a way to cluster them based on their similarities, that are the number and the kind of cuisines.
# 
# Briefly, after some steps of Data Cleaning and Data Exploration, it will be used a K-Means algorithm to extract the clusters, produce a map and make an argument on the final result.

# ## 2.3. Data Exploration

# To explore the data, it will be used the “Folium” a python library that can create interactive leaflet map using coordinate data.
# 
# A new entrepreneur will be able to choose the new location for your business based on two important premises:
# 
# • to find where the restaurants are located and to know their specialty, for example using the Foursquare API, or then 
# 
# • to use machine learning to bring out the general culinary trends and tastes of each neighborhood, for example the unsupervised learning method Clustering • to visualize the position of the various restaurants in the choosen district, for example with a Folium.

# # 3. Methodology

# ## 3.1. Importing libraries

# In[91]:


# First step, let's import the python libraries

import pandas as pd #library to handle data in vectorized manner
import numpy as np #library to convert the data in tabular form and perform data analysis
from pandas import json_normalize 
import json #libraryto handle JSOn files
from sklearn.cluster import KMeans

# Matplotlib and associated plotting modules
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


# Seaborn visualization library
import seaborn as sns

print('Libraries imported.')


# In[126]:


# Import libraries for web scraping and to handle requests
get_ipython().system('conda install -c conda-forge beautifulsoup4 --yes')

from bs4 import BeautifulSoup #library for pulling data from HTML and XML files
import requests #library to handle requests

# Import library for convert an address into latitude and longitude values
get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim

# Import library for map rendering
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes ')
import folium

get_ipython().system(' pip install yellowbrick')


# ## 3.2. Data Collection

# The city of Paris is divided into twenty arrondissements municipaux, administrative districts, more simply referred to as arrondissements. These are not to be confused with departmental arrondissements, which subdivide the larger French départements. The word "arrondissement", when applied to Paris, refers almost always to the municipal arrondissements listed below.
# 
# The data was extracted from the following website and imported as an excel spreadsheet.
# 
# https://opendata.paris.fr/explore/dataset/quartier_paris/table/?disjunctive.c_ar

# In[132]:


df_Paris=pd.read_excel('Documents\Pos-doutorado\Python\Coursera\Dados_Paris.xlsx', 'Plan1')
print('Dataframe shape:', df_Paris.shape)
df_Paris.head(6)


# In[133]:


df_Paris.shape


# In[134]:


# Rename the necessary columns 'Name', 'Car' and 'LAR' to 'Neighborhood', 'Arrondissement_Num' and 'French'
df_Paris.rename(columns={'NAME': 'Neighborhood', 'CAR': 'Arrondissement_Num',  'LAR': 'French_Name'}, inplace=True)

# Remove unnecessary columns.
df_Paris.drop(['NSQAR','CAR.1','CARINSEE','NSQCO','SURFACE', 'PERIMETRE' ], axis=1, inplace=True)

print('Dataframe shape:', df_Paris.shape)
df_Paris.head(6)


# In[135]:


# Sort data by arrondissement number for the sake of a nice look.
df_Paris.sort_values(by=['Arrondissement_Num'], inplace=True) 
df_Paris.reset_index(drop=True, inplace = True)

print('Dataframe shape:', df_Paris.shape)
df_Paris.head(6)


# ## 3.3 Getting Coordinates of Major Districts : Geopy Client

# **Get the coordinates of these 20 major neighbohood of Paris using geocoder class of Geopy client as follow:**

# In[138]:


# Retrieve the Latitude and Longitude for Paris
from geopy.geocoders import Nominatim 

address = 'Paris'

# User agent name is a drop of humor :D  
# Flânerie is the act of strolling, with all of its accompanying associations.
# The ability to wander detached from society with no other purpose than to be an acute observer of society.

geolocator = Nominatim(user_agent="Paris_explorer")

location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

print(f'The geographical coordinates of Paris (France) are {latitude} and {longitude}.')


# **Use python folium library to visualize geographic details of Paris and its neighbohood and create a map with boroughs superimposed on top. It was used latitude and longitude values to get the visual as below:**

# In[139]:


map_paris = folium.Map(location=[latitude, longitude], tiles='cartodbpositron', zoom_start=13)

# add circle shaped markers of districts
for lat, lng, label in zip(df_Paris['Latitude'], df_Paris['Longitude'], df_Paris['French_Name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=15,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_paris)  
    
map_paris


# # 3.4. Exploratory Data Analysis:

# **Use the Foursquare API to explore the Arrondissements of Paris (Neighborhoods)** 

# Define Foursquare Credentials and Version

# In[155]:


CLIENT_ID = '4RXRYTQPQVF0LDHKKBYNQOYIYTW414B03I2IJCMCRXR2QLQC' # your Foursquare ID
CLIENT_SECRET = '5MOFV0LBDKO5CZV3FHL2UUTAQQQ3AAPBZAPIHWOHFHVHBJTZ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 30

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# Let's explore the first neighborhood in our dataframe.

# In[156]:


# Explore the first Neighborhood in our dataframe.
df_Paris.loc[0, 'Neighborhood']


# In[157]:


# Get the Neighborhood's latitude and Longitude values.

neighborhood_latitude = df_Paris.loc[0, 'Latitude'] # Neighborhood latitude value
neighborhood_longitude = df_Paris.loc[0, 'Longitude'] # Neighborhood longitude value

neighborhood_name = df_Paris.loc[0, 'Neighborhood'] # Neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# # 3.5. Using Foursquare Location Data

# **Finally, let’s make use of Foursquare API and get the top 100 venues that are in Louvre within a radius of 500 meters.**

# In[160]:


limit = 100 # limit of number of venues returned by Foursquare API.
radius = 500 # Define radius. 500 is default number but good for start. 

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, CLIENT_SECRET, VERSION, neighborhood_latitude, neighborhood_longitude, radius, limit)

# Lets check whole URL
url


# In[161]:


# Send the GET request and examine the resutls
results = requests.get(url).json()

# We got a very big dictionary as result. After exploratory first look we can make some clarification for cleaner view.
results['response']['groups']


# ### Exploration of each venues in Paris

# Select necessary part of results, flatten and filtering

# In[163]:


# select necessary part of results and flatten
venues = results['response']['groups'][0]['items']
nearby_venues = json_normalize(venues) 

# filter only necessary columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = nearby_venues.loc[:, filtered_columns]


# **Define the function that extracts the category of the venues**

# In[164]:


# define a function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# **Structure the json file into a pandas dataframe**

# In[165]:


# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]


# In[166]:


print('Nearby_venues Dataframe shape:', nearby_venues.shape)
nearby_venues.head(4)


# In[167]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# **One request summary**

# In[168]:


# Let's see how many venues were found in 1eme Ardt within a radius of 500 meters.
print('{} venues were found and fetched from Foursquare.'.format(nearby_venues.shape[0]))


# In[169]:


print ('{} unique categories in Paris'.format(nearby_venues['categories'].value_counts().shape[0]))


# # Scale the algorithm to apply it to all neighborhoods

# Scale the algorithm to apply it to all neighborhoods

# In[170]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(name, lat, lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['French_Name', 
                  'Latitude', 
                  'Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[171]:


# Create a new dataframe and fill it
paris_venues = getNearbyVenues(names=df_Paris['French_Name'],
                                   latitudes=df_Paris['Latitude'],
                                   longitudes=df_Paris['Longitude'])


# In[172]:


print('Dataframe shape:',paris_venues.shape)
paris_venues.head(4)


# **Calculate how many unique venue categories there are in dataset for all neighborhoods.**

# In[173]:


print('There are {} unique venue categories.'.format(len(paris_venues['Venue Category'].unique())))


# ## Restaurants

# **Let's evaluate how many different restaurants represented among top restaurants**

# In[174]:


# Create a selected Dataframe to Concentrate Only on Restaurants 
paris_resto = paris_venues[paris_venues['Venue Category'].str.contains('Restaurant')].reset_index(drop=True)
paris_resto.index = np.arange(1, len(paris_resto)+1)


# In[175]:


print (paris_resto['Venue Category'].value_counts())


# In[176]:


print('There are {} uniques restaurants.'.format(len(paris_resto['Venue Category'].unique())))


# In[177]:


# create a dataframe of top 10 restaurant categories

paris_resto_top10 = paris_resto['Venue Category'].value_counts ()[0:10].to_frame(name='frequency')
paris_resto_top10.reset_index(inplace = True)

paris_resto_top10.iat[4, 0]='General Restaurant'
paris_resto_top10.rename(index=str, columns={"index": "Venue Category", "frequency": "Frequency"}, inplace=True)

paris_resto_top10


# In[232]:


# Draw graph 
s=sns.barplot(x="Venue Category", y="Frequency", data=paris_resto_top10)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('10 Most Frequently Occuring Restaurants in Paris', fontsize=20)
plt.xlabel("Venue Category - Restaurant", fontsize=16)
plt.ylabel ("Frequency", fontsize=16)
plt.savefig("Most_Freq_Restaurant.png", dpi=300)
fig = plt.figure(figsize=(18,10))
plt.show()


# In[179]:


paris_resto


# In[181]:


paris_resto_count = paris_resto.groupby(['French_Name'])['Venue Category'].apply(lambda x: x[x.str.contains('Restaurant')].count())
paris_resto_count=paris_resto_count.to_frame().reset_index()
paris_resto_count.rename(index=str, columns={"Venue Category": "Restaurants"}, inplace=True)
paris_resto_count.head(10)


# In[233]:


# Draw graph 
s=sns.barplot(x="French_Name", y="Restaurants", palette= 'Blues', data=paris_resto_count)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Restaurants Count by Neighborhood', fontsize=20)
plt.xlabel("Neighborhood", fontsize=15)
plt.ylabel ("Restaurants count", fontsize=15)
plt.savefig("Restaurants_count.png", dpi=300)
fig = plt.figure(figsize=(22,10))
plt.show();


# # Café, Bar and Creperies

# **Let's examine how many other different venues related to food and drink represented among all selected venues**

# In[183]:


# Set Category List. Drink & Beverages
drink_cat=['Coffee','Shop Bar','Wine Bar','Juice Bar','Beer Bar','Tea Room','Brasserie']
ffood_cat=['Café','Bistro','Pastry Shop','Bakery','Creperie','Salad Place','Pizza Place','Sandwich Place','Ice Cream Shop']


# In[184]:


# Create a selected Dataframe with Drink Category 
paris_drink = paris_venues[paris_venues['Venue Category'].str.contains('|'.join(drink_cat))].reset_index(drop=True)
paris_drink.index = np.arange(1, len(paris_drink)+1)
print (paris_drink['Venue Category'].value_counts())


# In[185]:


paris_drink_top10 = paris_drink['Venue Category'].value_counts()[0:10].to_frame(name='frequency')
paris_drink_top10.reset_index(inplace = True)
paris_drink_top10.rename(index=str, columns={"index": "Venue Category", "frequency": "Frequency"}, inplace=True)
paris_drink_top10


# In[234]:


# Draw graph

s=sns.barplot(x="Venue Category", y="Frequency", data=paris_drink_top10)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Drink and Beverages', fontsize=20)
plt.xlabel("", fontsize=15)
plt.ylabel ("Frequency", fontsize=15)
plt.savefig("Top_Drinks.png", dpi=300)
fig = plt.figure(figsize=(22,10))
plt.show()


# In[187]:


# Create a selected Dataframe with Café and Bistro Category 
paris_ffood = paris_venues[paris_venues['Venue Category'].str.contains('|'.join(ffood_cat))].reset_index(drop=True)
paris_ffood.index = np.arange(1, len(paris_ffood)+1)
print (paris_ffood['Venue Category'].value_counts())


# In[188]:


paris_ffood_top10 = paris_ffood['Venue Category'].value_counts()[0:10].to_frame(name='frequency')
paris_ffood_top10.reset_index(inplace = True)
paris_ffood_top10.rename(index=str, columns={"index": "Venue Category", "frequency": "Frequency"}, inplace=True)
paris_ffood_top10


# In[235]:


# Draw graph 
s=sns.barplot(x="Venue Category", y="Frequency", data=paris_ffood_top10)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('Bistro, Café and Creperie', fontsize=20)
plt.xlabel("", fontsize=15)
plt.ylabel ("Frequency", fontsize=15)
plt.savefig("Top_Bistro_Cafe.png", dpi=300)
fig = plt.figure(figsize=(22,10))
plt.show()


# ## 3.5 Normalize aggregated data and analyze each of the Neighborhoods

# **Make one hot encoding with 'Venue Category'**

# In[236]:


# one hot encoding
paris_norm = pd.get_dummies(paris_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
paris_norm['Neighborhood'] = paris_venues['French_Name'] 

# move neighborhood column to the first column
fixed_columns = [paris_norm.columns[-1]] + list(paris_norm.columns[:-1])
paris_norm = paris_norm[fixed_columns]


# In[191]:


# Check dataframe 
print('Dataframe shape:',paris_norm.shape)
paris_norm.head(6)


# **Group rows by neighborhood and take the mean of the frequency of occurrence of each category**

# In[192]:


paris_grouped = paris_norm.groupby('Neighborhood').mean().reset_index()

# Check dataframe 
print('Dataframe shape:', paris_grouped.shape)
paris_grouped.head(6)


# **Group rows by neighborhood and sum of occurrence of each category**

# In[193]:


# This data set would be usefull later on
paris_grouped_sum = paris_norm.groupby('Neighborhood').sum().reset_index()

# Check dataframe 
print('Dataframe shape:', paris_grouped_sum.shape)
paris_grouped_sum.head(6)


# **Quick look at each neighborhood with it's top 10 most common venues**

# In[194]:


top_common_venues = 10
for neigh in paris_grouped['Neighborhood'].sort_values():
    print("----- {} -------------".format(neigh,))
    temp = paris_grouped[paris_grouped['Neighborhood'] == neigh].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(top_common_venues))
    print('\n')


# **The top 10 venue categories for each neighborhood**

# **Shrink the data. Make new dataframe only from top common venues.**

# In[195]:


# Creating a function to extract top common venues.
def return_most_common_venues(row, top_common_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:top_common_venues]


# In[196]:


# Create the new dataframe. Arrondissements france name  and display the top 10 venues for each neighborhood
top_common_venues = 10
suffix = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(top_common_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, siffix[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
        
# create a new dataframe
paris_top_venues = pd.DataFrame(columns=columns)
paris_top_venues['Neighborhood'] = paris_grouped['Neighborhood']

for ind in np.arange(paris_grouped.shape[0]):
    paris_top_venues.iloc[ind, 1:] = return_most_common_venues(paris_grouped.iloc[ind, :], top_common_venues)

# final look of sorted data is
paris_top_venues


# This is a very useful results table that can provide at a glance information for all of the districts. Even once any conclusions are drawn further into the data workflow, we can refer back to this table for meaningful insights about the top categories of businesses in all the neighborhoods. Even without actual counts and numbers, it makes a great reference table.
# 
# As we can see, some districts have characteristic features. Some are dominated by classic restaurants, others, such as 15, are dominated by non-French restaurants (Italian, Indian and Thai cuisines). The characteristics we obtained as a result of a clear analysis emphasize and confirm the obvious characteristics of the central areas in which there are many points of interest and in which institutions focus mainly on tourists and the tourism industry.

# ## 3.6. Cluster Neighborhoods

# To analyze which neighborhood of Paris is good to open a new restaurant, we can use a K-means clustering: a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.
# 
# So the first step is identify the best “K” using a famous analytical approach: the elbow method.
# 
# Let's see:

# In[226]:


from yellowbrick.cluster import KElbowVisualizer
paris_clustering = paris_grouped.drop('Neighborhood', 1)

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,11))

visualizer.fit(paris_clustering)        # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# From the plot up here, I can easily say that the best K is 6.
# 
# Finally, we can try to cluster the neighborhood based on the venue categories and use K-Means clustering. The 6 clusters are partitioned based on similar type of restaurants that belong to neighborhoods.
# 

# **Run k-means to cluster the neighborhood into 6 clusters**

# In[197]:


clust_num = 6
paris_grouped_clust = paris_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(paris_grouped_clust)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# **Let's create a new dataframe that includes the cluster as well as the top 10 venues for each arrondissement.**

# In[201]:


# reserved
paris_top_venues_labeled = paris_top_venues


# In[203]:


# add clustering labels
#paris_top_venues_labeled.insert(0, 'Cluster', kmeans.labels_)
paris_top_venues_labeled.rename(columns={'Neighborhood':'French_Name'}, inplace=True)

paris_merged = df_Paris

# merge paris_top_venues_labeled with paris_arr initial data to add latitude/longitude for each neighborhood
paris_merged = paris_merged.join(paris_top_venues_labeled.set_index('French_Name'), on='French_Name')


# In[204]:


print('Dataframe shape:', paris_merged.shape)
paris_merged.head(6)


# **Finally, let's visualize the resulting clusters**

# In[212]:


list_resto_count=paris_resto_count['Restaurants'].to_list()
list_neigh=paris_resto_count['French_Name'].to_list()


# In[213]:


list_resto_count.append(2)
list_neigh.append('12eme Ardt')


# In[237]:


# create map
paris_map_resto = folium.Map(location=[latitude,longitude], tiles='cartodbpositron', zoom_start=13)

# set color scheme for the clusters
x = np.arange(clust_num)
ys = [i + x + (i*x)**2 for i in range(clust_num)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
for lat, lon, poi, cluster in zip(paris_merged['Latitude'], 
                                  paris_merged['Longitude'], 
                                  paris_merged['French_Name'], 
                                  paris_merged['Cluster']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=list_resto_count[list_neigh.index(poi)]*3,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(paris_map_resto)
       
paris_map_resto


# **Examine the Clusters**

# Now, we can examine each cluster and determine the discriminating venue categories that distinguish each cluster. Based on the defining categories, we can assign a name to each uster.

# **Cluster 1.**

# In[216]:



paris_merged.loc[paris_merged['Cluster'] == 0, paris_merged.columns[[1] + list(range(5, paris_merged.shape[1]))]]


# **Cluster 2**

# In[217]:



paris_merged.loc[paris_merged['Cluster'] == 1, paris_merged.columns[[1] + list(range(5, paris_merged.shape[1]))]]


# **Cluster 3**

# In[218]:



paris_merged.loc[paris_merged['Cluster'] == 2, paris_merged.columns[[1] + list(range(5, paris_merged.shape[1]))]]


# **Cluster 4**

# In[219]:



paris_merged.loc[paris_merged['Cluster'] == 3, paris_merged.columns[[1] + list(range(5, paris_merged.shape[1]))]]


# **Cluster 5**

# In[220]:


paris_merged.loc[paris_merged['Cluster'] == 4, paris_merged.columns[[1] + list(range(5, paris_merged.shape[1]))]]


# In[221]:


df_Paris


# # 4. Results & Discussion

# This data analysis shows us that each neighborhood has its particularity. Some are dominated by French Restaurant such as, Pantheon, Buttes-Montmartre and  Opera, others, as Vaugirard, are dominated by Italian Cuisine.
# 
# The characteristics we obtained as a result of a clear analysis emphasize and confirm the obvious characteristics of the central areas in which there are many points of interest and in which institutions focus mainly on tourists and the tourism industry.

# The analysis shows that there are areas where there is a balanced number of restaurants, cafes and other catering venues. The result emphasizes the actual and general characteristics of the districts in the clusters. The opening of a restaurant in this area is quite reasonable.
# 
# The infrastructure of the districts already meets the needs of people for food and leisure. People are already considering these areas for lunch, dinner, meetings and evening rest. Any venue that opens in these areas will benefit from the status of the place and the habits of the people.
# 
# The high-level business question remains: whether to use the strengths of the place and compete in a classic, tight environment or to offer yourself in a less crowded niche. You can open as a classic restaurant, or if the analysis shows that small, fast food restaurants are in demand - open as a snackbar with chosen cuisine
# 
# This is where the part of the analysis that shows what type of institution dominates in the area can help. Accordingly, what is most in demand and what is the nature of consumers in this area. This analysis within this project is quite superficial, it shows the basic methods and opportunities. You can refine your search criteria and improve your analysis if the task is more thorough and specific. But already now the preliminary analysis and especially clustering has revealed characteristic groups of areas on which it is possible to concentrate more specifically. So, for further consideration, I would choose three clusters.
# 
# Cluster 1 and 4 as HiClass with Hi Cuisine Restaurant, Hotels and Bars, and Cluster 2 as universally interesting cluster who definitely deserves a detailed analysis.
# We have made inferences from the data in making the location recommendations, but that is exactly the point. There is no right or wrong answer or conclusion for the task at hand. The job of data analysis here is to steer a course for the location selection of new restaurant (1) to meet the criteria of being in neighbourhoods that are lively with abundant leisure venues, and (2) to narrow the search down to just a few of the main areas that are best suited to match the criteria.

# In[ ]:




