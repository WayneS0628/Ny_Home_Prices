README

New York Housing Prices Prediction

This project is a simple linear regression model built using Python to analyze and predict housing prices over time. It is designed to work with any Zillow Home Value Index (ZHVI) dataset, as long as the geography is set to the state level. By default, this project extracts and analyzes New York State data, but users can modify the preprocessing step to select a different state.

Features

- Dataset Compatibility: Works with any state-level ZHVI dataset (e.g., All Homes, Single-Family, Condo, 1-Bedroom, Multi-Bedroom).
- Automated Preprocessing: Converts wide-format datasets into a structured format for regression.
- Customizable State Selection: Defaults to New York State, but users can change this in the filtering step.
- Machine Learning Model: Implements Simple Linear Regression to identify trends in housing prices.
- Forecasting Capabilities: Predicts future home values based on historical trends.
- Visualizations: Uses Matplotlib to display actual vs. predicted prices with regression trendlines.

How to Get the Zillow Dataset

- Go to the Zillow Research Data Page: https://www.zillow.com/research/data/
- Select “Home Value Index (ZHVI)” under “Browse Data Categories.”
- Choose the desired dataset (e.g., All Homes, Single-Family, Condo, etc.).
- Set the geography filter to “State” to ensure compatibility with this project.
- Download the dataset as a CSV file.
- Place the CSV file in the project directory for preprocessing and analysis.

Changing the State for Analysis

By default, the preprocessing script filters for New York State. To analyze a different state, update the following line in the script:

  ny_data = dataset[dataset["RegionName"] == "New York"]

Feel free to clone and use as you desire.

Future Improvements

- Add more features such as interest rates, inflation, or population growth.
- Implement polynomial regression for better accuracy.
- Use time-series forecasting methods for improved trend analysis.
