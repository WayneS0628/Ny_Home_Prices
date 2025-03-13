import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('/Users/waynesimmonsjr/TrainingArc/Test_2/State_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')

# Extractig NY price data and dates

# Step 1: Filter for only New York
ny_data = dataset[dataset["RegionName"] == "New York"] # This extracts 
# all the rows where the "RegionName" column is "New York" so it creates a pandas dataframe with only 1 row
# since theres only 1 state name New York lol, and a bunch of columns with their headers

# print(ny_data.head()) # uncomment to test it

# Step 2: Select only the columns that contain dates
date_columns = dataset.columns[5:]  # This assumes that the 6th column onward contains dates 
# (which is true in this dataset)
# So this essentially creates a pandas index object which just contains a list of all
# The column names headers so not the values in the clumns just the columns names
# which in our case is just a list of dates in string format in a pandas index list

# print(date_columns) # uncomment to test it

# Step 3: Melt the dataset (reshape)
ny_melted = ny_data.melt(id_vars=["RegionName"], value_vars=date_columns, var_name="Date", value_name="AvgPrice")
# now we apply melt to ny_data which is a pandas dataframe object

# id_vars is set to regions becaus emelt requires at least one unchanged column from 
# the dataframe in order to work

# value_vars would take a list of index objects or columns names to look for in the dataframe.
# We have to specify what columns so that melt can extract the values under those columns and 
# convert to a seperate column. Bceuase the ny_data dataframe has some columns at the beginning 
# that are not dates (eg. RegionID, SizeRank, RegionType) we have to make sure we only 
# tell melt to select the columns that are dates. So technically we could list all the columns 
# names we want like 2000-01-31,2000-02-29,2000-03-31,2000-04-30,...,2025-01-31 but that would 
# take too long. Thankfully we already have a list of all the columns that are dates in the
# date_columns variable so we can just pass that in as the value_vars parameter

# var_name is the name of the new column that will contain the column names from the original 
# dataframe that were melted 

# value_name is the name of the new column that will contain the values from the original
# dataframe that were melted 

# Note that the values (prices in our case) are pulled from the columsn that melts finds 
# associated with the column names so for example if for value_var we passed in only 2000-01-31
# then the values in the AvgPrice column would be the values under the 2000-01-31 column in the
# original dataframe which is only one price value. But since we passed in all the date columns
# the values in the AvgPrice column will be all the price values in the original dataframe
# that were under their specific date columns

# print(ny_melted.head()) # uncomment to test it

# Step 4: Drop the "RegionName" column since it's redundant now
ny_melted = ny_melted.drop(columns=["RegionName"]) 

# This drops the region name column because it not needed anymoe we only included it originally
# because melt requires at least one unchanged column from the dataframe in order to work 
# uncomment the most recent above print statement to see the new dataframe with the region name 
# column before we removed it. Uncomment the print statement below to see the new dataframe 
# without the region name column (which is the data frame we're going to use)
print(ny_melted.head())




ny_melted["Date"] = pd.to_datetime(ny_melted["Date"]) # This converts the "Date" column to a datetime object
# We do this because the "Date" column is currently a string object but we want it to be a datetime object
# We want it to be a datetime object becaus linear regression models can't take in string objects as input
# And datetime object will be easier to work with when we plot the data


reference_date = pd.Timestamp("2000-01-31")  # The starting point (Month 0) for the x-axis
# This creates a pandas timestamp object that represents the date "2000-01-31"
# We'll use this as the starting point for the x-axis when we plot the data
# This is because the x-axis will represent the number of months since "2000-01-31"

# Convert dates to period month format and get integer representation
ny_melted["Months_Since_2000"] = (
    ny_melted["Date"].dt.to_period("M").astype(int) -
    pd.Period("2000-01", freq="M").to_timestamp().to_period("M").ordinal
)

# print(ny_melted.head()) # uncomment to test it

# Define feature (X) and target (y)
X = ny_melted["Months_Since_2000"].values.reshape(-1, 1)  # Independent variable (Months Since 2000)
y = ny_melted["AvgPrice"].values.reshape(-1, 1)  # Dependent variable (Average Home Price)

# Handle missing data (though there likely isn’t any)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the simple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set results
y_pred = regressor.predict(X_test)

# Visualize training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Months Since Jan, 2000 vs Avg Home Price in NY (Training Set)")
plt.xlabel("Months Since Jan, 2000")
plt.ylabel("Avg Home Price in NY")
plt.show()

# Visualize test set results
plt.scatter(X_test, y_test, color="red", label="Actual Prices")
plt.scatter(X_test, y_pred, color="green", label="Predicted Prices", marker="x")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Months Since Jan, 2000 vs Avg Home Price in NY (Test Set)")
plt.xlabel("Months Since Jan, 2000")
plt.ylabel("Avg Home Price in NY")
plt.show()

# Calculate and print the R-squared score
print("R-squared score:", r2_score(y_test, y_pred))

# Predict future home prices
future_X = np.array([[312]])  # Example: Month 312 (January 2026)
predicted_price = regressor.predict(future_X)
print("Predicted home price in Jan 2026:", predicted_price[0][0])

future_X = np.array([[400]])  # Example: Month 400 (May 2033)
predicted_price = regressor.predict(future_X)
print("Predicted home price in May 2033:", predicted_price[0][0])