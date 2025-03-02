# Housing Price Prediction with Linear Regression

This program is designed to predict housing prices based on a dataset provided in CSV format. It utilizes a simple linear regression model to establish a relationship between the features in the dataset (e.g., number of bedrooms, square footage) and the target variable (housing price).

**Purpose:**

The primary purpose of this program is to:

* **Provide a basic example of housing price prediction:** It demonstrates how to load data, train a linear regression model, and evaluate its performance.
* **Offer a starting point for more complex analyses:** Users can adapt and extend this code to explore more advanced machine learning techniques and improve prediction accuracy.
* **Enable quick analysis of housing datasets:** This program allows users to quickly gain insights into the relationship between features and housing prices within their own datasets.
* **Visualize prediction results:** The program generates a scatter plot to visually compare actual versus predicted housing prices, providing a clear understanding of the model's performance.

**How it Works:**

1.  **Data Loading:** The program reads a CSV file containing housing data. It automatically detects the delimiter used in the file.
2.  **Model Training:** It splits the data into training and testing sets and trains a linear regression model on the training data.
3.  **Prediction and Evaluation:** The trained model predicts housing prices for the testing data, and the program calculates performance metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score.
4.  **Visualization:** Finally, the program generates a scatter plot showing the relationship between actual and predicted prices, and saves it as a PNG file.

**Who is it for?**

This program is suitable for:

* Individuals interested in learning about basic machine learning concepts.
* Data analysts looking for a quick and easy way to analyze housing datasets.
* Anyone who wants to experiment with linear regression for price prediction.
* Students learning about linear regression.
