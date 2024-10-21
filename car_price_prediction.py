import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pickle

# Data Processing Function
def process_data(city_files):
    datasets = []
    for file in city_files:
        try:
            dataset = pd.read_excel(file, engine='openpyxl')  # Read Excel files
            datasets.append(dataset)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            continue

    if datasets:
        final_dataset = pd.concat(datasets, ignore_index=True)

        # Handling Missing Values
        for col in final_dataset.select_dtypes(include=['float64', 'int64']).columns:
            final_dataset[col].fillna(final_dataset[col].mean(), inplace=True)

        for col in final_dataset.select_dtypes(include=['object']).columns:
            final_dataset[col].fillna(final_dataset[col].mode()[0], inplace=True)

        # Standardizing Data Formats
        if 'distance' in final_dataset.columns:
            final_dataset['distance'] = final_dataset['distance'].str.replace(' kms', '').astype(int)

        # Encoding Categorical Variables
        final_dataset = pd.get_dummies(final_dataset, drop_first=True)
        
        return final_dataset
    else:
        st.error("No valid datasets found.")
        return None

# Model Training Function
def train_model(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        st.write(f"{name}: Mean CV MSE: {-scores.mean()}")

    # Hyperparameter Tuning Example
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Model Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R^2': r2_score(y_test, y_pred)
    }

# Streamlit Application
def main():
    st.title("Car Price Prediction")

    # Upload files
    uploaded_files = st.file_uploader("Upload your city datasets", type='xlsx', accept_multiple_files=True)
    
    if uploaded_files:
        # Validate that all uploaded files are Excel files
        for file in uploaded_files:
            if not file.name.endswith('.xlsx'):
                st.error(f"{file.name} is not an Excel file. Please upload valid Excel files.")
                return

        # Process data
        final_dataset = process_data(uploaded_files)

        if final_dataset is not None and 'target' in final_dataset.columns:
            X = final_dataset.drop('target', axis=1)  # Features
            y = final_dataset['target']  # Target variable

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = train_model(X_train, y_train)

            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)
            st.write(f"Model Performance: {metrics}")

            # Save model
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)

            # Input features for prediction
            st.subheader("Input Features for Prediction")
            input_data = []

            for column in X.columns:
                if np.issubdtype(X[column].dtype, np.number):
                    value = st.number_input(f"{column}", value=0.0)
                    input_data.append(value)
                else:
                    value = st.selectbox(f"{column}", options=final_dataset[column].unique())
                    input_data.append(value)

            if st.button("Predict"):
                input_data = np.array([input_data])  # Prepare input with the correct feature order
                prediction = model.predict(input_data)
                st.write(f"Predicted Price: {prediction[0]}")

if __name__ == "__main__":
    main()
