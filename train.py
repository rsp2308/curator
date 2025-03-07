import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
import os
from urllib import request

def download_data_files():
    """Download the required data files if they don't exist"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # OECD BLI data
    oecd_bli_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/oecd_bli_2015.csv"
    oecd_bli_path = "data/oecd_bli_2015.csv"
    if not os.path.exists(oecd_bli_path):
        print(f"Downloading {oecd_bli_path}...")
        request.urlretrieve(oecd_bli_url, oecd_bli_path)
    
    # GDP per capita data
    gdp_per_capita_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv"
    gdp_per_capita_path = "data/gdp_per_capita.csv"
    if not os.path.exists(gdp_per_capita_path):
        print(f"Downloading {gdp_per_capita_path}...")
        request.urlretrieve(gdp_per_capita_url, gdp_per_capita_path)
    
    return oecd_bli_path, gdp_per_capita_path

def prepare_country_stats(oecd_bli, gdp_per_capita):
    """Prepare and merge the data for the linear regression model"""
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    # Join the two datasets
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, 
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    
    # Keep only the Life satisfaction and GDP per capita columns
    return full_country_stats[["Life satisfaction", "GDP per capita"]]

# Main program
try:
    # Download the data if needed and get file paths
    oecd_bli_path, gdp_per_capita_path = download_data_files()
    
    # Load the data
    oecd_bli = pd.read_csv(oecd_bli_path, thousands=',')
    gdp_per_capita = pd.read_csv(gdp_per_capita_path, thousands=',', 
                               delimiter='\t', encoding='latin1', na_values="n/a")
    
    # Prepare the data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]
    
    # Visualize the data
    plt.figure(figsize=(8, 6))
    country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', 
                      figsize=(8, 6), grid=True)
    plt.title("Life Satisfaction vs. GDP per Capita")
    
    # Add country names as annotations
    for country, x, y in zip(country_stats.index, 
                           country_stats["GDP per capita"], 
                           country_stats["Life satisfaction"]):
        plt.annotate(country, xy=(x, y), xytext=(5, -5), 
                    textcoords='offset points', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')  # Save visualization
    plt.show()
    
    # Select a linear model
    model = sklearn.linear_model.LinearRegression()
    # Uncomment to try a K-Nearest Neighbors model instead:
    # model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)  # Fixed n_neighbors type
    
    # Train the model
    model.fit(X, y)
    
    # Plot the model's predictions
    X_new = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_predict = model.predict(X_new)
    plt.figure(figsize=(8, 6))
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.plot(X, y, "bo", label="Data")
    plt.xlabel("GDP per capita (USD)")
    plt.ylabel("Life satisfaction")
    plt.legend(loc="lower right")
    plt.title("Life Satisfaction Prediction Model")
    plt.savefig('model_predictions.png')  # Save predictions
    plt.show()
    
    # Make a prediction for Cyprus
    X_cyprus = [[22587]]  # Cyprus' GDP per capita
    print(f"Predicted life satisfaction in Cyprus: {model.predict(X_cyprus)[0][0]:.2f}")
    
    # Print model parameters
    if isinstance(model, sklearn.linear_model.LinearRegression):
        print(f"Linear Model Equation: Life satisfaction = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.6f} × GDP per capita")
        print(f"Slope: {model.coef_[0][0]:.6f}")
        print(f"Intercept: {model.intercept_[0]:.2f}")

except Exception as e:
    print(f"An error occurred: {e}") 