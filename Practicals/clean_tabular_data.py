# %%
import numpy as np
import pandas as pd
import plotly
import missingno as msno
import os
# %%
os.getcwd()
# %%
######## CLEAN THE IMAGES AND CSV FILES TO DATAFRAMES
## import Images CSV files
images_df = pd.read_csv('Images.csv')
images_df = images_df.drop(columns='Unnamed: 0', axis=1)
# %%
## Preview the dataframe
images_df.head()
# %%
## Check for the number of null values
images_df.isnull().sum()
# %%
## Import Product CSV Files
products_df = pd.read_csv('Products.csv', lineterminator='\n')
products_df = products_df.drop(columns='Unnamed: 0', axis=1)
# %%
## Preview and check for null values
products_df.head()
products_df.isnull().sum()
# %%
## Test the products dataframe for missing data
msno.heatmap(products_df)
# %%
products_df['price'].unique()
# %%

### Clean products price column - removing the £ symbol and comma
price_regex_pattern = {'£': "",
                       ',': ""}

products_df['price'] = products_df['price'].replace(price_regex_pattern, regex=True)

# %%
products_df['price'].sort_values(ascending=False)
# %%
## Instanstiate the price column as a float
products_df['price'] = products_df['price'].astype(float)
# %%
products_df['price'].sort_values(ascending=False)
# %%
products_df.head()
# %%
## Split the categories using the / terminator and take the first category as main_category
products_df['main_category'] = products_df['category'].str.split(' /', expand=True)[0]
# %%
## CHeck that the categories are unique
products_df['main_category'].unique()
# %%
######## Extracting Labels for Classification
pd.get_dummies(products_df['main_category'])

# %%
products_df['main_category_code'] = products_df['main_category'].astype('category').cat.codes
# %%
products_df
# %%
