import matplotlib.pyplot as plt
import pandas_datareader as pdr

# # # Data science steps

# # # 1. Data gathering

# Set the beginning and end of the historical data
start_date = '1990-01-01'
end_date = '2023-01-23'

# Download the VIX (volatility index, or the "fear index") data into a Pandas dataframe
vix = pdr.DataReader(
    'VIXCLS',  # data name
    'fred',  # data source
    start_date, end_date,  # dates
)

# Store data:
# - homogeneous dataset for effective storage -> arrays
# - heterogeneous data or need to work in a tabular manner -> dataframes

# Print the first and last five observations of the dataframe
print(vix.head())
#             VIXCLS
# DATE
# 1990-01-02   17.24
# 1990-01-03   18.19
# 1990-01-04   19.22
# 1990-01-05   20.11
# 1990-01-08   20.26

print(vix.tail())
#             VIXCLS
# DATE
# 2023-01-17   19.36
# 2023-01-18   20.34
# 2023-01-19   20.52
# 2023-01-20   19.85
# 2023-01-23   19.81

# # # 2. Data preprocessing

# Calculate the number of NaN values
count_nan = vix['VIXCLS'].isnull().sum()

# Print the result
print('Number of nan values in the VIX dataframe:', count_nan)
# Number of nan values in the VIX dataframe: 292

# How the missing values can be handled:
# 1. Delete the cell that contains the missing value;
# 2. Assume that the missing cell is equal to the previous cell;
# 3. Calculate a mean or a median of the cells around the empty value.

# Drop the NaN values from the rows
vix = vix.dropna()

# Take the differences in an attempt to make the data stationary
vix = vix.diff(periods=1, axis=0)
#             VIXCLS
# DATE
# 1990-01-02     NaN
# 1990-01-03    0.95
# 1990-01-04    1.03
# 1990-01-05    0.89
# 1990-01-08    0.15
# ...            ...
# 2023-01-17   -0.13
# 2023-01-18    0.98
# 2023-01-19    0.18
# 2023-01-20   -0.67
# 2023-01-23   -0.04
#
# [8333 rows x 1 columns]

# Drop the first row of the dataframe
vix = vix.iloc[1:, :]
#             VIXCLS
# DATE
# 1990-01-03    0.95
# 1990-01-04    1.03
# 1990-01-05    0.89
# 1990-01-08    0.15
# 1990-01-09    1.94
# ...            ...
# 2023-01-17   -0.13
# 2023-01-18    0.98
# 2023-01-19    0.18
# 2023-01-20   -0.67
# 2023-01-23   -0.04
#
# [8332 rows x 1 columns]

# # # 3. Data exploration

# Calculate the mean of the dataset
mean = vix["VIXCLS"].mean()
# Print the result
print('The mean of the dataset:', mean)
# The mean of the dataset: 0.0003084493518963043

# # # 4. Data visualization

# Plot the latest 250 observations in darkgrey with a label
plt.plot(vix[-250:], color='darkgrey', linewidth=1.5, label='Change in VIX')
# Plot a red dashed horizontal line that is equal to mean
plt.axhline(y=mean, color='red', linestyle='dashed')
# Add a grid to facilitate the visual component
plt.grid()
# Add the legend function so it appears with the chart
plt.legend()
# # Show in a window
# plt.show()
plt.savefig('example-01-1--data-science-for-vix.svg')
plt.close()

# # # 5. Data analysis, 6. Data interpretation / prediction are skipped for now to study later
