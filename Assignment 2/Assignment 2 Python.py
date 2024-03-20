# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:30:03 2023

@author: Sibo Ding
"""

import pandas as pd

#%% 1
airquality = pd.read_csv('airquality.csv')
airquality.head() # First 6 rows

# 1)
airquality_long = pd.melt(airquality, id_vars=['Month', 'Day'], 
                          value_vars=['Ozone', 'Solar.R', 'Wind', 'Temp'],
                          var_name='Measurement', value_name='Value')
airquality_long.head()

# 2)
airquality_unite = airquality_long.copy()
airquality_unite['Date'] = airquality_unite['Month'].astype(str)\
    + '-' + airquality_unite['Day'].astype(str)
airquality_unite.drop(columns=['Month', 'Day']).head()

#%% 2
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'customer_name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
    'city': ['New York', 'San Francisco', 'Boston', 'Seattle', 'Chicago']
    })
orders = pd.DataFrame({
    'customer_id': [1, 1, 2, 2, 2, 3, 3, 4, 5],
    'order_id': [101, 102, 201, 202, 203, 301, 302, 401, 501],
    'order_amount': [100, 200, 150, 75, 225, 300, 225, 175, 250]
    })

# If there are multiple matches between Left and Right,
# all combinations of the matches are returned.
left_join_df = pd.merge(customers, orders, how='left', on='customer_id')
left_join_df
right_join_df = pd.merge(customers, orders, how='right', on='customer_id')
right_join_df
inner_join_df = pd.merge(customers, orders, how='inner', on='customer_id')
inner_join_df
full_join_df = pd.merge(customers, orders, how='outer', on='customer_id')
full_join_df

# Return all rows from Left where there are matching values in Right,
# keeping just columns from Left.
semi_join_df = customers[customers['customer_id'].isin(orders['customer_id'])]
semi_join_df
# Return all rows from Left where there are not matching values in Right,
# keeping just columns from Left.
anti_join_df = customers[~customers['customer_id'].isin(orders['customer_id'])]
anti_join_df

#%% 3
df1 = pd.DataFrame({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
df2 = pd.DataFrame({'id': [3, 4, 5], 'value': ['c', 'd', 'e']})

concat_df = pd.concat([df1, df2], axis=0)
union_df = concat_df.drop_duplicates().reset_index(drop=True)
union_df
intersect_df = concat_df[concat_df.duplicated()].reset_index(drop=True)
intersect_df

setdiff_df_1_2 = pd.concat([df1, intersect_df], axis=0)\
    .drop_duplicates(keep=False).reset_index(drop=True)
setdiff_df_1_2
setdiff_df_2_1 = pd.concat([df2, intersect_df], axis=0)\
    .drop_duplicates(keep=False).reset_index(drop=True)
setdiff_df_2_1

#%% 4
import requests

# https://stackoverflow.com/questions/39710903/pd-read-html-imports-a-list-rather-than-a-dataframe
url = 'https://www.imdb.com/chart/top/'
req = requests.get(url)
tab_list = pd.read_html(req.text)
movies = tab_list[0] # Get the first table

movies = movies[['Rank & Title', 'IMDb Rating']]\
    .rename(columns={'Rank & Title': 'title', 'IMDb Rating': 'rating'})
# Extract release year within ( )
movies['release_year'] = movies['title'].str.extract('\((\d{4})\)')
# Remove everything after "  (" and before ".  "
movies['title'] = movies['title']\
    .str.replace('\s{2}\((.*)', '', regex=True)\
    .str.replace('(.*)\.\s{2}', '', regex=True)
movies.head(10)

#%% 5
import re

# \d: digit, {}: numbers of occurrence
re.findall('\d{3}-\d{3}-\d{4}', 
           'Please call us at 123-456-7890 or 555-555-5555.')

# \w: word character; +: pne or more occurrences
re.findall('\w+@\w+.\w+', 
           'Contact us at info@example.com or support@example.com.')

re.sub('\w+://\w+.\w+.\w+', 'URL', 
       'Check out our website at https://www.example.com ' + \
           'and our blog at https://blog.example.com.')

#%% 6
date_data = pd.DataFrame({'date_time': ['2023-02-22 7:30:15', 
                                        '2023-02-23 12:15:30', 
                                        '2023-02-24 23:59:59']})

date_data['date'] = pd.to_datetime(date_data['date_time']).dt.date
date_data['time'] = pd.to_datetime(date_data['date_time']).dt.time
date_data

#%% 7a
books = pd.read_csv('books.csv')
books.head() # First 5 rows
books.shape # Number of observations, Number of variables
books.columns # Names of variables

#%% 7b
books.dropna(axis=0, subset='author')\
    .groupby('author').count()\
    .sort_values(by='title', ascending=False).head()

# Shakespeare, William has the most publications (326).

#%% 7c
shakespeare_books = books[
    (books['author'] == 'Shakespeare, William') & (books['language'] == 'en')]
shakespeare_books.head()

#%% 7d
from urllib.request import urlopen

shakespeare_hamlet = shakespeare_books[shakespeare_books['title'] == 'Hamlet']\
    [['gutenberg_id', 'title', 'author']].head(1)

# Download "Hamlet" text
# https://realpython.com/python-web-scraping-practical-introduction/
gutenberg_id = 1787
url = f'https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt'
page = urlopen(url)
hamlet_text_str = page.read().decode('utf-8')

# Convert text to dataframe
hamlet_text = pd.DataFrame(
    [i for i in hamlet_text_str.split('\n')], columns=['text'])[4:]
hamlet_text['gutenberg_id'] = gutenberg_id

# Lastly
hamlet_data = pd.merge(
    shakespeare_hamlet, hamlet_text, how='left', on='gutenberg_id')\
    .dropna(axis=0, subset='text')
hamlet_data['text'] = hamlet_data['text'].str.lower()

hamlet_data.head()

#%% 7e
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# First
hamlet_sentiments = pd.read_csv('sentiments_afinn.csv')
# To get afinn sentiment value, can also use package and iterate each word
# from afinn import Afinn
# Afinn().score('love')

# Second
# https://towardsdatascience.com/5-simple-ways-to-tokenize-text-in-python-92c6804edfc4
hamlet_token = pd.DataFrame(
    word_tokenize(hamlet_text_str.lower()), columns=['word'])

# Third
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
stop_words = pd.DataFrame(stopwords.words('english'), columns=['word'])

# Fourth
hamlet_join = hamlet_token[~hamlet_token['word'].isin(stop_words['word'])]\
    .merge(hamlet_sentiments, how='inner', on='word')
hamlet_join['n'] = 1

# Fifth
hamlet_words = hamlet_join\
    .groupby(['word', 'value']).count().sort_values(by='n', ascending=False)\
    .reset_index()
hamlet_words.head(10)

#%% 7f
import matplotlib.pyplot as plt

hamlet_top_words = hamlet_words.loc[hamlet_words.groupby('value')['n'].idxmax()]\
    .sort_values(by='n', ascending=False)

plt.bar(hamlet_top_words['word'], hamlet_top_words['value'])
plt.title('Most common words for each sentiment value in Hamlet')
plt.xlabel('Most common words')
plt.ylabel('Corresponding sentiment value')
