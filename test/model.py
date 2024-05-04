# Importing libraries
import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt')

# Load data
df = pd.read_csv("scraped_data.csv")

# Data preprocessing
column_to_drop = 'Image'
if column_to_drop in df.columns:
    df = df.drop(columns=[column_to_drop])

df.drop_duplicates(subset='Product Name', inplace=True)
df['Sold'] = df['Sold'].astype(str)

df['Sold'] = df['Sold'].apply(lambda x: re.search(r'౹\s*([\d\s]+)\s*\+', x).group(1).replace(' ', '') if pd.notna(x) and re.search(r'౹\s*([\d\s]+)\s*\+', x) else 0).astype(int)
df['Sold'] = df['Sold'].combine_first(df['Sold'].astype(str).apply(lambda x: re.search(r'(\d+)', x).group(1) if pd.notna(x) and re.search(r'(\d+)', x) else 0).astype(int))

zero_sold_count = (df['Sold'] == 0).sum()
df['Sold'] = np.where(df['Sold'] == 0, 0, 1)

df['Price'] = df['Price'].str.replace('MAD', '').str.replace(',', '').str.replace(' ', '').astype(float)
df['Reviews'] = df['Reviews'].str.replace(' Reviews', '').astype(str)
df['Rating'] = df['Rating'].str.extract('(\d+\.?\d*)').astype(float)
df['Reviews'] = df['Reviews'].str.extract('(\d+\.?\d*)').astype(float)

# Data visualization
sns.countplot(x='Sold', data=df)
plt.title('Distribution of Sold/Not Sold')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
sns.boxplot(x='Sold', y='Price', data=df, ax=axes[0])
axes[0].set_title('Relationship between Sold and Price')

sns.boxplot(x='Sold', y='Rating', data=df, ax=axes[1])
axes[1].set_title('Relationship between Sold and Rating')

plt.show()

sold_values = df[df['Sold'] == 1]['Price']
not_sold_values = df[df['Sold'] == 0]['Price']
t_statistic, p_value = ttest_ind(sold_values, not_sold_values, equal_var=False)
print(f'T-test p-value for Price: {p_value}')

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
sns.boxplot(x='Sold', y='Rating', data=df)
plt.title('Relationship between Sold and Rating')
plt.show()

df['ProductNameLength'] = df['Product Name'].apply(lambda x: len(str(x)))

# NLP preprocessing
df['Product Name'] = df['Product Name'].astype(str)
df['Product Name Tokens'] = df['Product Name'].apply(lambda x: word_tokenize(x) if pd.notnull(x) else [])
stemmer = PorterStemmer()
df['Product Name Stemmed'] = df['Product Name Tokens'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

# Machine learning model
df['Sold'] = df['Sold'].astype(bool)

vectorizer = CountVectorizer(max_features=10, stop_words='english')
keywords_matrix = vectorizer.fit_transform(df['Product Name'].astype('str'))
keywords_df = pd.DataFrame(keywords_matrix.toarray(), columns=vectorizer.get_feature_names_out())

category_columns = pd.DataFrame()
category_column_name = 'Category'  # Replace with the actual column name

for category in df[category_column_name].unique():
    category_columns[f'{category_column_name}_{category}'] = (df[category_column_name] == category).astype(int)

df = pd.concat([df, category_columns, keywords_df], axis=1)

X = df[['Price', 'Rating', 'Reviews', 'ProductNameLength'] + list(category_columns.columns) + list(keywords_df.columns) + ['Sold']]
df_cleaned = X.dropna()

y = df_cleaned['Sold'].astype(int)
X = df_cleaned.drop('Sold', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')



import pickle

# Save the trained model to a file
with open('your_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)