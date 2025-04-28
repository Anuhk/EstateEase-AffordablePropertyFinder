import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the data
df = pd.read_csv('Mumbai_updated_realistic.csv')

# Let's simulate 'Area' and 'Bedrooms' if missing
import numpy as np
if 'Area' not in df.columns:
    df['Area'] = np.random.randint(400, 1200, size=len(df))  # area in sqft
if 'Bedrooms' not in df.columns:
    df['Bedrooms'] = np.random.randint(1, 4, size=len(df))  # 1 to 3 bedrooms

# Features and Target
X = df[['Location', 'Society', 'Area', 'Bedrooms']]
y = df['Price']

# Preprocessing (Location and Society are categorical)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Location', 'Society'])
    ],
    remainder='passthrough'  # Area and Bedrooms go as-is
)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data (optional, for testing accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'house_price_predictor.pkl')

print("Model trained and saved as 'house_price_predictor.pkl' ✅")
