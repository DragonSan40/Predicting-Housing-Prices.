import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file path
file_path = "housing.csv"
output_graph_path = "housing_price_prediction.png"

# Check if file exists
if not os.path.exists(file_path):
    print("Error: File not found. Please check the file path.")
else:
    try:
        # Load raw file content to inspect
        with open(file_path, 'r') as f:
            first_line = f.readline()
        print("Raw first line of dataset:", first_line)

        # Load the dataset with auto-detected delimiter
        possible_delimiters = [',', '\\t', ';', '|', '\\s+']
        df = None
        for delim in possible_delimiters:
            df_try = pd.read_csv(file_path, delimiter=delim, engine="python")
            if df_try.shape[1] > 1:
                df = df_try
                print(f"Delimiter detected: '{delim}'")
                break

        if df is None:
            raise ValueError("Unable to detect the correct delimiter. Please check the dataset format.")

        # Print dataset info to check structure
        print("Dataset Preview:")
        print(df.head())
        print("\nDataset Shape:", df.shape)

        # Define features (X) and target (y)
        X = df.iloc[:, :-1]  # All columns except last
        y = df.iloc[:, -1]  # Last column as target

        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print evaluation metrics
        print("\nModel Performance:")
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("R-squared Score (R²):", r2)

        # Plot actual vs predicted values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  # Diagonal line
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Housing Prices")
        plt.savefig(output_graph_path)  # Save the graph as a PNG file
        plt.show()
        print(f"Graph saved as {output_graph_path}")

        #Print column names.
        print("\nFeatures used:", list(X.columns))
        print("Target used:", y.name)

    except Exception as e:
        print("Error loading dataset:", str(e))