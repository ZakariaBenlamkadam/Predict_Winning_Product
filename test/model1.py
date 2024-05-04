import pandas as pd
import re 
import numpy as np

df = pd.read_csv("scraped_data.csv")

# Assuming df is your DataFrame
column_to_drop = 'Image'

if column_to_drop in df.columns:
    df = df.drop(columns=[column_to_drop])

# Assuming df is your DataFrame
df.drop_duplicates(subset='Product Name', inplace=True)

# Ensure 'Sold' column is treated as strings
df['Sold'] = df['Sold'].astype(str)

# Extract the number of sold items after the "౹" symbol
df['Sold'] = df['Sold'].apply(lambda x: re.search(r'౹\s*([\d\s]+)\s*\+', x).group(1).replace(' ', '') if pd.notna(x) and re.search(r'౹\s*([\d\s]+)\s*\+', x) else 0).astype(int)

# If 'Sold' column is still NaN (for cases like '14 sold'), extract the number
df['Sold'] = df['Sold'].combine_first(df['Sold'].astype(str).apply(lambda x: re.search(r'(\d+)', x).group(1) if pd.notna(x) and re.search(r'(\d+)', x) else 0).astype(int))

zero_sold_count = (df['Sold'] == 0).sum()
zero_sold_count

df['Sold'] = np.where(df['Sold'] == 0, 0, 1)

# Assuming df is your DataFrame
df['Price'] = df['Price'].str.replace('MAD', '').str.replace(',', '').str.replace(' ', '').astype(float)

# Assuming df is your DataFrame
df['Reviews'] = df['Reviews'].str.replace(' Reviews', '').astype(str)


# Convert 'Rating' to float (handle errors='coerce' to handle non-numeric values)
df['Rating'] = df['Rating'].str.extract('(\d+\.?\d*)').astype(float)

# Convert 'Reviews' to numeric (handle errors='coerce' to handle non-numeric values)
df['Reviews'] = df['Reviews'].str.extract('(\d+\.?\d*)').astype(float)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Assuming df is your DataFrame
# Replace 'Sold' with the actual column name if it's different

# Distribution of the target variable
sns.countplot(x='Sold', data=df)
plt.title('Distribution of Sold/Not Sold')
plt.show()

# Explore relationships with numerical features (Price, Rating)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Relationship with Price
sns.boxplot(x='Sold', y='Price', data=df, ax=axes[0])
axes[0].set_title('Relationship between Sold and Price')

# Relationship with Rating
sns.boxplot(x='Sold', y='Rating', data=df, ax=axes[1])
axes[1].set_title('Relationship between Sold and Rating')

plt.show()

# Statistical tests for significant differences (t-test)
sold_values = df[df['Sold'] == 1]['Price']
not_sold_values = df[df['Sold'] == 0]['Price']

t_statistic, p_value = ttest_ind(sold_values, not_sold_values, equal_var=False)
print(f'T-test p-value for Price: {p_value}')

# Convert 'Rating' to float (handle errors='coerce' to handle non-numeric values)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Visualize the relationship between 'Rating' and 'Sold' after conversion
sns.boxplot(x='Sold', y='Rating', data=df)
plt.title('Relationship between Sold and Rating')
plt.show()

# Add 'ProductNameLength' column
df['ProductNameLength'] = df['Product Name'].apply(lambda x: len(str(x)))




import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

# Convert 'Product Name' to string
df['Product Name'] = df['Product Name'].astype(str)

# Tokenization
df['Product Name Tokens'] = df['Product Name'].apply(lambda x: word_tokenize(x) if pd.notnull(x) else [])

# Stemming
stemmer = PorterStemmer()
df['Product Name Stemmed'] = df['Product Name Tokens'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

# Check the updated DataFrame
df[['Product Name', 'Product Name Tokens', 'Product Name Stemmed']].head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Assuming df is your original DataFrame

# Convert 'Sold' to boolean
df['Sold'] = df['Sold'].astype(bool)

# Extract keywords from 'Product Name'
vectorizer = CountVectorizer(max_features=10, stop_words='english')
keywords_matrix = vectorizer.fit_transform(df['Product Name'].astype('str'))

# Create a DataFrame from the keyword matrix
keywords_df = pd.DataFrame(keywords_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Initialize an empty DataFrame for category columns
category_columns = pd.DataFrame()

# Iterate over unique values in the column containing category information
category_column_name = 'Category'  # Replace with the actual column name
for category in df[category_column_name].unique():
    # Create a binary column indicating the presence of the category
    category_columns[f'{category_column_name}_{category}'] = (df[category_column_name] == category).astype(int)

# Concatenate the new category columns and keywords features with the original DataFrame
df = pd.concat([df, category_columns, keywords_df], axis=1)

# Include the relevant columns in X (features)
X = df[['Price', 'Rating', 'Reviews', 'ProductNameLength'] + list(category_columns.columns) + list(keywords_df.columns) + ['Sold']]

# Drop rows with missing values
df_cleaned = X.dropna()

# Define your target variable
y = df_cleaned['Sold'].astype(int)  # Convert boolean to integer

# Drop the target variable from X
X = df_cleaned.drop('Sold', axis=1)

# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model with an increased max_iter
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')