# %%
import pandas as pd
import numpy as np
import os
# %%
## Create 2 dictionaries
dict_1 = {'London': [51.509865,-0.118092]} 
dict_2 = {'Paris': [48.864716, 2.349014]}
# %%
my_list = [dict_1, dict_2]
# %%
my_list
# %%
len(my_list)
# %%
for i in range(len(my_list)):
    #print(list(my_list[i].keys())[0])
    print(list(my_list[i].keys())[0],':', list(my_list[i].values())[0][0])
# %%

## Create a tupe of 5 elements and print each element

my_tuple = (1,2,3,4,5)
# %%
my_tuple
# %%
type(my_tuple)
# %%
len(my_tuple)
# %%
for i in range(len(my_tuple)):
    print(f'element',i, 'is', my_tuple[i])
# %%
print(my_tuple[3])
# %%
'''
Create a list of 3 dictionaries, where each dictionary contains the keys "name", "age" and "favourite_colour".
Assign appropriate values to each key in each dictionary.
Using nested for loops or list comprehension, create a new dictionary where the value of each "name" key is a key. 
The values in the new dictionary should be a dictionary of the other two key-value pairs from each original dictionary. 
As an example, the new dictionary should key-values pairs like this: "Sarah" : {"age": 25, "favourite_colour", "blue"}


'''

dict_1 = {'name': 'rupert', 'age': 39, 'favourite_colour': 'blue'}

dict_2 = {'name': 'leo', 'age': 7, 'favourite_colour': 'red'}

dict_3 = {'name': 'melanie', 'age': 40, 'favourite_colour': 'pink'}
# %%

dict_list = [dict_1, dict_2, dict_3]
# %%
list(dict_list[i].keys())[0]
# %%
## instantiate a new dictionary
neuer_dictionary = {}
for i in range(len(dict_list)):
    for key, value in dict_list[i].items():
        neuer_dictionary[list(dict_list[i].values())[0]] = {list(dict_list[i].keys())[1]: list(dict_list[i].values())[1], list(dict_list[i].keys())[2]: list(dict_list[i].values())[2]}
# %%
neuer_dictionary

# %%
type(neuer_dictionary)
# %%

### importing a yaml file
os.getcwd()
import json
import yaml
from pandas.io.json import json_normalize

# %%
with open('yaml_example.yaml') as yamlfile:
    configuration = yaml.safe_load(yamlfile)
# %%
yamlfile.close()
# %%

## Converting the yaml file to json
with open('yaml_json.json', 'w') as jsonfile:
    json.dump(configuration, jsonfile)
jsonfile.close()

# %%
yamljson =  json.load(open('yaml_json.json'))
# %%
type(yamljson)

yamljson.keys()
# %%

## Normalise the json based on the record path - testing
pd.json_normalize(yamljson, 'Animals')
# %%

## This is a function to convert a yaml file to json to a pandas df through specifying a particular record
def convert_yaml_to_df(yaml_file_location, record):
    with open(yaml_file_location) as yamlfile:
        configuration = yaml.safe_load(yamlfile)
    yamlfile.close()

    with open('yaml_json.json', 'w') as jsonfile:
        json.dump(configuration, jsonfile)
    jsonfile.close()

    yaml_to_json = json.load(open('yaml_json.json'))

    df = pd.json_normalize(yamljson, record)

    return df
# %%
convert_yaml_to_df('yaml_example.yaml', 'Person')
# %%
'''
Use the Salaries.csv file calculate:
What is the ratio between people in the fire department over people in the police department
What is the mean salary of the police department? Use the "BasePay" column.
What is the mean salary of the fire department? Use the "BasePay" column.

'''

## Read in salaries CSV

salaries = pd.read_csv('Salaries.csv')
# %%
salaries.describe()
# %%
salaries.info()
# %%
salaries.head()
# %%
## Extract from Job title - Fire Department or Police Department

## Count the number of employees across both categories

salaries['IsPolice'] = salaries['JobTitle'].str.contains('POLICE')

salaries['IsFire'] = salaries['JobTitle'].str.contains('FIRE')
# %%
salaries.head()
# %%
## Number of police officers
len(salaries[salaries['IsPolice'] == True])

## Number of fire officers
len(salaries[salaries['IsFire'] == True])
# %%
test_dict = {}
# %%
len(test_dict.keys()) == 0
# %%
salaries = pd.read_csv('Salaries.csv')
# %%
salaries.head()
# %%
number_unique_jobs = len(salaries['JobTitle'].unique())
# %%
number_unique_jobs
# %%
salaries[['name_1', 'name_2', 'name_3', 'name_4']] = salaries['EmployeeName'].str.split(" ",expand=True)
# %%
salaries['name_1'].value_counts()

# %%

def long_surname_finder(string):
    if len(string) > 6:
        return string
    else:
        return 'short name'

# %%

# %%
