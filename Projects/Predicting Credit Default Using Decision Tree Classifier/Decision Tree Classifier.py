# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:28:48 2024

@author: Riya Ohri
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Function to fetch and load the German Credit Data
def load_german_credit_data():
    """
    Loads the German Credit Data from OpenML.

    Returns:
        df (DataFrame): The German Credit DataFrame.
    """
    from sklearn.datasets import fetch_openml
    german = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')  # Set parser='auto' to silence FutureWarning
    df = german.frame
    print("Loaded DataFrame Columns:")
    print(df.columns)
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst Few Rows:")
    print(df.head())
    return df

# Function to preprocess the data
def preprocess_data(df):
    """
    Preprocesses the German Credit Data.

    Steps:
        - Rename target column to 'Default'.
        - Encode target variable.
        - Separate features and target.
        - Convert known categorical numeric columns to 'category' dtype.
        - Identify categorical and numerical columns.
        - Handle missing values.
        - Encode categorical variables using One-Hot Encoding.
        - Convert the processed features into a DataFrame.

    Args:
        df (DataFrame): The raw German Credit DataFrame.

    Returns:
        X (DataFrame): Feature matrix after preprocessing.
        y (Series): Target vector.
    """
    # Rename target column for clarity
    df = df.rename(columns={'class': 'Default'})

    # Replace target labels with binary values and convert to int
    df['Default'] = df['Default'].map({'good': 0, 'bad': 1}).astype(int)

    # Verify that mapping was successful
    if df['Default'].isnull().any():
        raise ValueError("Mapping of 'Default' column resulted in NaN values. Check the input data.")

    # Separate features and target
    X = df.drop('Default', axis=1)
    y = df['Default']

    # Convert known categorical numeric columns to 'category' dtype
    categorical_numeric_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ]
    for col in categorical_numeric_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Debugging: Print categorical and numerical columns
    print(f"Categorical Columns: {categorical_cols}")
    print(f"Numerical Columns: {numerical_cols}\n")

    # Handle missing values if any (German Credit Data typically has no missing values)
    # If there are missing values, uncomment the following lines to impute them
    # imputer = SimpleImputer(strategy='mean')
    # X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    # Encode categorical variables using One-Hot Encoding if there are any
    if categorical_cols:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
            ],
            remainder='passthrough'  # Keep numerical columns as is
        )

        X_processed = preprocessor.fit_transform(X)

        # Get feature names from OneHotEncoder
        ohe = preprocessor.named_transformers_['cat']
        ohe_features = ohe.get_feature_names_out(categorical_cols)
        all_features = list(ohe_features) + numerical_cols
    else:
        print("No categorical columns to encode. Proceeding with numerical features only.")
        X_processed = X.values
        all_features = numerical_cols

    # Convert the preprocessed features back to a DataFrame for better interpretability
    X_processed = pd.DataFrame(X_processed, columns=all_features)

    # Debugging: Print shapes and feature names
    print(f"Shape of Feature Matrix: {X_processed.shape}")
    print(f"Feature Names: {all_features}\n")

    return X_processed, y   

# Function to train Decision Tree Classifier using Gini Impurity
def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    """
    Trains a Decision Tree Classifier using Gini Impurity.

    Args:
        X_train (DataFrame): Training feature matrix.
        y_train (Series): Training target vector.
        max_depth (int, optional): The maximum depth of the tree. Defaults to None.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        clf (DecisionTreeClassifier): The trained Decision Tree model.
    """
    clf = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

# Function to evaluate the model
def evaluate_model(clf, X_test, y_test):
    """
    Evaluates the Decision Tree model on the test set.

    Args:
        clf (DecisionTreeClassifier): The trained Decision Tree model.
        X_test (DataFrame): Testing feature matrix.
        y_test (Series): Testing target vector.

    Returns:
        y_pred (ndarray): Predicted target values.
    """
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}\n')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    return y_pred

# Function to visualize the Decision Tree
def visualize_decision_tree(clf, feature_names, class_names, max_depth=3):
    """
    Visualizes the Decision Tree using Graphviz.

    Args:
        clf (DecisionTreeClassifier): The trained Decision Tree model.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        max_depth (int, optional): The maximum depth for the visualization. Defaults to 3.
    """
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")  # Saves as decision_tree.pdf
    graph.view()

# Function to plot feature importances
def plot_feature_importances(clf, feature_names):
    """
    Plots the feature importances of the Decision Tree.

    Args:
        clf (DecisionTreeClassifier): The trained Decision Tree model.
        feature_names (list): List of feature names.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

# Function for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

def tune_decision_tree(X_train, y_train):
    """
    Performs hyperparameter tuning on the Decision Tree Classifier using GridSearchCV.

    Args:
        X_train (DataFrame): Training feature matrix.
        y_train (Series): Training target vector.

    Returns:
        best_clf (DecisionTreeClassifier): The best Decision Tree model after tuning.
    """
    param_grid = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    clf = DecisionTreeClassifier(criterion='gini', random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}\n")

    best_clf = grid_search.best_estimator_
    return best_clf

# Function for Cross-Validation
from sklearn.model_selection import cross_val_score

def cross_validate_model(clf, X, y, cv=5):
    """
    Performs cross-validation on the Decision Tree model.

    Args:
        clf (DecisionTreeClassifier): The Decision Tree model.
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        scores (ndarray): Array of cross-validation scores.
    """
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}\n")
    return scores

# Main function to execute the workflow
def main():
    try:
        # Load data
        df = load_german_credit_data()
        print("Data Loaded Successfully.\n")
        
        # Preprocess data
        X, y = preprocess_data(df)
        print("Data Preprocessing Completed.\n")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Training Set Size: {X_train.shape[0]} samples")
        print(f"Testing Set Size: {X_test.shape[0]} samples\n")
        
        # Train Decision Tree Classifier
        clf = train_decision_tree(X_train, y_train, max_depth=5, random_state=42)
        print("Decision Tree Training Completed.\n")
        
        # Evaluate the model
        y_pred = evaluate_model(clf, X_test, y_test)
        
        # Plot Feature Importances
        feature_names = X.columns.tolist()
        plot_feature_importances(clf, feature_names)
        
        # Visualize the Decision Tree
        class_names = ['Good', 'Bad']
        visualize_decision_tree(clf, feature_names, class_names, max_depth=3)
        print("Decision Tree Visualization Saved as 'decision_tree.pdf'.\n")
        
        # Hyperparameter Tuning
        best_clf = tune_decision_tree(X_train, y_train)
        print("Best Decision Tree Training Completed.\n")
        
        # Evaluate the best model
        y_pred_best = evaluate_model(best_clf, X_test, y_test)
        
        # Visualize the Best Decision Tree
        visualize_decision_tree(best_clf, feature_names, class_names, max_depth=3)
        print("Best Decision Tree Visualization Saved as 'decision_tree.pdf'.\n")
        
        # Cross-Validation
        cross_validate_model(best_clf, X, y, cv=5)
        
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
