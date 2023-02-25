# %%
import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio
import numpy as np
import missingno

# %%
## Set plotly theme
pio.templates.default = "plotly_dark"
pd.options.plotting.backend = 'plotly'

# %%
## Instructions
'''
Download the dataset from the following URL: https://aicore-files.s3.amazonaws.com/Data-Eng/Building_Permits.csv
Load it into a DataFrame
Observe the name of the columns and the corresponding dtype
Check how many missing values each column has
Calculate the percentage of missing values for each column
Using the missingno library, plot the nullity matrix. Do you see any pattern in data missingness?
Create a copy of your DataFrame (call it df_filled_backwards). Use the back fill method to fill in the missing values
Create a copy of your DataFrame (call it df_filled_forwards). Use the forward fill method to fill in the missing values
Create a copy of your DataFrame (call it df_filled_linearly). Use the linear fill method to fill in the missing values
For the three new dataframes, plot the distribution of both 'Street Number Suffix' and 'Zipcode'

'''

bpermits_df = pd.read_csv('https://aicore-files.s3.amazonaws.com/Data-Eng/Building_Permits.csv')
# %%
bpermits_df.info()
# %%
bpermits_df.head()
# %%
## Calculate the percentage of missing values for each column
for col in bpermits_df.columns:
    print(col, ((bpermits_df[col].isnull().sum())/len(bpermits_df)*100))
# %%
missingno.matrix(bpermits_df)
# %%
