import pickle
import numpy as np

with open('your_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

price = float(input("Enter the product price: "))
rating = float(input("Enter the product rating: "))
reviews = float(input("Enter the number of reviews: "))
product_name_length = int(input("Enter the length of the product name: "))

print("Enter the category (choose one):")
print("1. home, 2. clothes, 3. garden, 4. accessories, 5. gym, 6. sports, 7. camping, 8. phones, 9. computers, 10. health, 11. technology, 12. beauty, 13. electronic")
category_input = input("Enter the category number: ")
category_columns = ['Category_home', 'Category_clothes', 'Category_garden', 'Category_accessories', 'Category_gym', 'Category_sports', 'Category_camping', 'Category_phones', 'Category_computers', 'Category_health', 'Category_technology', 'Category_beauty', 'Category_electronic']
category_values = [0] * len(category_columns)
category_values[int(category_input) - 1] = 1

keyword_features = []
for keyword_column in ['computer', 'fitness', 'garden', 'men', 'mini', 'outdoor', 'pc', 'phone', 'sports', 'women']:
    keyword_input = input(f"Enter value for {keyword_column}: ")
    keyword_features.append(float(keyword_input))  # Assuming the features are numeric

input_features = np.array([[price, rating, reviews, product_name_length] + category_values + keyword_features])


prediction = model.predict(input_features)

print(f"The predicted outcome for the product is: {prediction}")
