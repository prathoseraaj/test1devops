import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv('cleaned_unique_jiomart_devops_.csv')  # Update this with your actual CSV filename

# Data preprocessing
# Drop any rows with missing values for simplicity
data = data.dropna(subset=['sale_price', 'market_price', 'category', 'type', 'rating'])

# Encode categorical variables ('category', 'type') into numerical values
data['category'] = data['category'].astype('category').cat.codes
data['type'] = data['type'].astype('category').cat.codes

# Prepare feature (X) and target (y) variables
# Features: 'category', 'type', 'rating' (we'll predict sale_price)
X = data[['category', 'type', 'rating']]  # Adjust based on relevant columns
y = data['sale_price']  # Target: sale_price

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'jiomart_sale_price_model.pkl')
