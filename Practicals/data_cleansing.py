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
## using the backfill method for na values
df_filled_backwards = bpermits_df.fillna(method='bfill')
# %%
## using the forward fill method for na values
df_filled_forwards = bpermits_df.fillna(method='ffill')
# %%
## using the linear interpolate method
df_filled_linearly = bpermits_df.interpolate(method='linear')
# %%
from plotly.subplots import make_subplots
# For the three new dataframes, plot the distribution of both 'Street Number Suffix' and 'Zipcode'

fig_bw_street = px.histogram(df_filled_backwards, "Street Suffix")
fig_bw_zip = px.histogram(df_filled_backwards, "Zipcode")

fig_ff_street = px.histogram(df_filled_forwards, "Street Suffix")
fig_ff_zip = px.histogram(df_filled_forwards, "Zipcode")

fig_li_street = px.histogram(df_filled_linearly, "Street Suffix")
fig_li_zip = px.histogram(df_filled_linearly, "Zipcode")

fig = make_subplots(rows=2, cols=3, shared_xaxes=False, subplot_titles=('Backwards: Street & Zip', 'Forwards: Street & Zip', 'Linear: Street & Zip'))

fig.add_trace(fig_bw_street['data'][0], row=1, col=1)
fig.add_trace(fig_bw_zip['data'][0], row=2, col=1)
fig.add_trace(fig_ff_street['data'][0], row=1, col=2)
fig.add_trace(fig_ff_zip['data'][0], row=2, col=2)
fig.add_trace(fig_li_street['data'][0], row=1, col=3)
fig.add_trace(fig_li_zip['data'][0], row=2, col=3)

# %%
fig.show()
# %%
