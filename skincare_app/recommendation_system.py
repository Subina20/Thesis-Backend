import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset (replace 'custom_data.csv' with the actual filename)
df = pd.read_csv('custom_data.csv', encoding='ISO-8859-1')

# Drop rows with missing values
df.dropna(inplace=True)

# Create dictionaries to map string labels to integer codes for skin type and acne type
skin_type_mapping = {
    'combination': 1,
    'oily': 2,
    'dry': 3,
    'sensitive': 4
}

acne_type_mapping = {
    'inflammatory': 1,
    'comedonal': 2,
    'cystic': 3,
    'nodular': 4
}

# Encode categorical variables
categorical_cols = ['skin_type', 'acne_type', 'brand']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# Content-Based Filtering Model Training
# Step 1: Create TF-IDF vectors for the 'name' and 'ingredients' features
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['name'] + ' ' + df['ingredients'])

# Step 2: Calculate cosine similarity between the 'name' and 'ingredients' vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get top N similar items based on content similarity
def get_top_similar_indices(indices, N=5):
    # Sort the indices based on similarity scores
    similar_indices = indices.argsort()[::-1]
    # Return top N similar indices
    return similar_indices[1:N+1]

# Function to get skin care product recommendations along with their brands based on user input
def get_skin_care_recommendations(age_str, skin_type_str, acne_type_str, N=5):
    try:
        # Convert string age input to an integer
        age = int(age_str)
    except ValueError:
        return None, None  # Return None if the age input is not a valid integer
    
    # Check if the user's age is within the valid range based on the dataset
    min_age = df['age'].min()
    max_age = df['age'].max()
    if age < min_age or age > max_age:
        return None, None
    
    skin_type = skin_type_mapping.get(skin_type_str.lower())
    acne_type = acne_type_mapping.get(acne_type_str.lower())
    
    if skin_type is None or acne_type is None:
        return None, None  # If the provided skin type or acne type is not valid, return None
    
    # Filter products based on user-provided age, skin type, and acne type
    filtered_products = df[(df['age'] <= age) & (df['skin_type'] == skin_type) & (df['acne_type'] == acne_type)]
    
    if filtered_products.empty:
        return None, None  # If no products match the criteria, return None
    
    # Get the indices of filtered products
    product_indices = filtered_products.index
    
    # Calculate average similarity scores for each product
    product_sim_scores = cosine_sim[product_indices].mean(axis=0)
    
    # Get the top N similar products
    top_similar_indices = get_top_similar_indices(product_sim_scores, N)
    
    # Get the top N recommended product names
    top_product_names = df['name'].iloc[top_similar_indices].values
    
    # Get the brand indices for the top N recommended products
    top_brand_indices = df['brand'].iloc[top_similar_indices].values
    
    # Get the brand names based on the brand indices
    brand_mapping = {i: label_encoders['brand'].inverse_transform([i])[0] for i in top_brand_indices}
    top_brands = [brand_mapping[i] for i in top_brand_indices]
    
    return top_product_names, top_brands

# Example usage of the recommendation system with user input from Flutter
# The inputs from Flutter will be in string format, so we pass them accordingly.
user_age_str = '30'  # Replace with user-provided age from Flutter
user_skin_type_str = "oily"  # Replace with user-provided skin type from Flutter
user_acne_type_str = "cystic"  # Replace with user-provided acne type from Flutter
num_recommendations = 8

product_recommendations, product_brands = get_skin_care_recommendations(user_age_str, user_skin_type_str, user_acne_type_str, num_recommendations)

if product_recommendations is not None:
    # Convert the results to a list of dictionaries
    results = []
    for product_name, brand in zip(product_recommendations, product_brands):
        result_dict = {
            "product_name": product_name,
            "brand": brand
        }
        results.append(result_dict)
        
    # Convert the results to JSON format and print it
    json_results = json.dumps(results, indent=2)
    print(json_results)
else:
    print("No matching products found based on the provided criteria.")
