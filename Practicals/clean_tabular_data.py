# %%
import numpy as np
import pandas as pd
import plotly
import missingno as msno
import os
# %%
os.getcwd()
# %%
## import Product and Images CSV files
images_df = pd.read_csv('Images.csv')
images_df = images_df.drop(columns='Unnamed: 0', axis=1)
# %%
images_df.head()
# %%
images_df.isnull().sum()
# %%
products_df = pd.read_csv('Products.csv', lineterminator='\n')
# %%
products_df = products_df.drop(columns='Unnamed: 0', axis=1)
# %%
products_df.head()
# %%
products_df.isna().sum()
# %%
msno.matrix(products_df)
# %%
products_df['product_name'].unique()
# %%
products_df.shape
# %%
products_df.info()
# %%
products_df
# %%
### Clean products price column
price_regex_pattern = {'£': "",
                       ',': ""}

products_df['price'] = products_df['price'].replace(price_regex_pattern, regex=True)

# %%
products_df['price'] = products_df['price'].astype(float)
# %%
products_df.info()
# %%
