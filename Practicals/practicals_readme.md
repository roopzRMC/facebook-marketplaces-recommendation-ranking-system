# Practicals

## Pandas and Data and Cleaning

The 2 csv files and images folder were copied from the VM EC2 instance via an ssh connection

> Ingesting the data

`Products.csv` needed to be imported using the `pd.read_csv` method with an additional `line_terminator` argument accounting for '\n'

`Images.csv` was in the standard CSV encoding and required no additional arguments

The images csv file however had an unecessary row number colums which was dropped after import



> Checking for null values

Leveraging `.isna().sum()` across both dataframes yielded no missing values

To double check, a missingno matrix was instantiated

> Data Types and validation

The Product Dataframe price column had been imported as an object rather than a numeric column. This is due to the presence of the £ and , symbols present in rows

To remove these erroneous characters, a dictionary was created containing the character to find and its equivalent replacement

```python
price_regex_pattern = {'£': "",
                       ',': ""}

products_df['price'] = products_df['price'].replace(price_regex_pattern, regex=True)

```

Once removed, the column could then be converted to a float type

> Encoding the master / main category


The category hierarchy was embedded in the category column using the '/' as a separator

Using `str.split` with the separator split the categories in to their constituent parts. By chaining `str.get` it was possible to extract the first Master category.


```python
products_df['category'].str.split(' /').str.get(0).unique()

```

Once these were extracted, it was first required that the datatype of the new column be converted from object to categorical. This allows for `cat.codes` to be called to convert each of these categories to a unique integer value starting at 0

```python
products_df['main_category_code'] = products_df['main_category'].astype('category').cat.codes
```

For the decoder - as cat codes works sequentially in alphabetical ascending order a dictionary was constructed that maps the key (cat code) to the original master category

```python
unique_cat_codes = list(products_df['main_category'].astype('category').cat.codes.unique())
unique_cats = list(products_df['main_category'].unique())
cat_decoder = {}
for i in range(len(unique_cat_codes)):
    cat_decoder[unique_cat_codes[i]] = unique_cats[i]
```

>Merging Products and Images data

Using `pd.merge` both datasets were merged on id and product id with cat codes remaining as the training data csv file

> Cleaning the images

The collected images are in varying sizes and also vary between between having all 3 RGB channels as well as grayscale.

For ingest in to a Machine Learning model, these need to be standardised both in terms of size and channels

The `resize_image` function takes an image and a dimension value and returns a modified image

This function is leveraged in a for loop

The for loop uses the size of the images directory (ie # images contained) and both takes each image in the directory, resizes it to 300 px by 300 px and converts it to grayscale)

Each new image is saved to a cleaned_images directory