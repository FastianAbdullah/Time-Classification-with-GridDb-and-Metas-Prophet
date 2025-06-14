### Key Points
- The blog post focuses on time-series forecasting, not classification, using GridDB and Prophet, despite the title suggesting classification.
- It covers loading, cleaning, and storing daily minimum temperature data in GridDB, then forecasting with Prophet, achieving an MAE of 2.09 and RMSE of 2.67.
- The structure follows a provided example, with sections for prerequisites, data insertion, forecasting, and conclusion, using code snippets from the notebook.

### Introduction
This blog post guides you through forecasting daily minimum temperatures using GridDB, a NoSQL database for time-series data, and the Prophet model by Facebook. It’s based on a Jupyter notebook, and while the title mentions "classification," the content is about forecasting, not classification, which we’ll clarify as we go.

### Prerequisites
You’ll need these tools installed:
- GridDB Python client: `pip install griddb-python` ([GridDB Python Client](https://pypi.org/project/griddb-python/))
- Prophet: `pip install prophet` ([Prophet Documentation](https://facebook.github.io/prophet/))
- Pandas, Matplotlib, Seaborn, Scikit-learn: Install via pip with their respective commands ([Pandas Documentation](https://pandas.pydata.org/), [Matplotlib Documentation](https://matplotlib.org/), [Seaborn Documentation](https://seaborn.pydata.org/), [Scikit-learn Documentation](https://scikit-learn.org/))
Ensure GridDB is running locally for this example.

---

### Survey Note: Detailed Analysis of Time-Series Forecasting with GridDB and Prophet

#### Introduction and Context
Time-series analysis is a vital technique for predicting future values based on historical data, with applications ranging from weather forecasting to financial analysis. This blog post, titled "Time-Series Classification with GridDb and Meta Prophet Model," is based on a Jupyter notebook that focuses on time-series forecasting, specifically predicting daily minimum temperatures. The title's mention of "classification" appears to be a misnomer, as the notebook does not involve classification tasks but rather demonstrates a forecasting workflow using GridDB and the Prophet model, developed by Facebook (now Meta). This discrepancy is noted for clarity, and the content will proceed with the forecasting approach as outlined in the notebook.

The dataset used is daily minimum temperatures, and the goal is to forecast future values using historical trends and seasonal patterns. GridDB, a high-performance NoSQL database optimized for time-series data, is used for data storage and management, while Prophet is chosen for its ability to handle seasonality and trends effectively. The blog post is structured to mirror the format of a provided example at [GridDB's blog](https://griddb.net/en/blog/time-series-classification-with-amazon-chronos-model-with-griddb/), which outlines a similar workflow for forecasting electricity production using Amazon Chronos and GridDB. Here, we adapt that structure to focus on the Prophet model and the temperature dataset, ensuring a comprehensive guide for readers.

#### Prerequisites and Setup
To replicate this workflow, specific libraries must be installed. The prerequisites section lists the following tools, each with installation commands and relevant documentation:

- **GridDB Python client**: Essential for interacting with GridDB, installed via `pip install griddb-python` ([GridDB Python Client](https://pypi.org/project/griddb-python/)).
- **Prophet**: A forecasting tool by Facebook, installed via `pip install prophet` ([Prophet Documentation](https://facebook.github.io/prophet/)).
- **Pandas**: For data manipulation, installed via `pip install pandas` ([Pandas Documentation](https://pandas.pydata.org/)).
- **Matplotlib**: For plotting, installed via `pip install matplotlib` ([Matplotlib Documentation](https://matplotlib.org/)).
- **Seaborn**: For enhanced visualizations, installed via `pip install seaborn` ([Seaborn Documentation](https://seaborn.pydata.org/)).
- **Scikit-learn**: For model evaluation metrics, installed via `pip install scikit-learn` ([Scikit-learn Documentation](https://scikit-learn.org/)).

Additionally, a GridDB cluster must be running and accessible. The example assumes a local setup, but users can configure it for their environment using GridDB’s documentation, ensuring scalability for larger datasets or real-time applications.

#### Inserting Time Series Data Into GridDB
The workflow begins with loading and cleaning the dataset, which contains daily minimum temperatures stored in a CSV file at '/home/ali/griddb_experiments/daily-temperature-dataset/daily-minimum-temperatures.csv'. The data includes two columns: 'Date' and 'Daily minimum temperatures'.

First, the dataset is loaded into a pandas DataFrame for initial inspection:

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('/home/ali/griddb_experiments/daily-temperature-dataset/daily-minimum-temperatures.csv')
print(data.head())
```

The 'Daily minimum temperatures' column initially contains string values, which are cleaned by extracting numeric values using a regular expression:

```python
# Convert temperature column to numeric
data['Daily minimum temperatures'] = pd.to_numeric(data['Daily minimum temperatures'].str.extract('(\d+\.\d+)').squeeze(), errors='coerce')
```

Next, the data is stored in GridDB, a process that involves connecting to the cluster and creating a time-series container. The connection is established as follows:

```python
from gs import GridStore

# Connect to GridDB
gridstore = GridStore("localhost", 10040, "admin", "admin", "test")

# Create a time series container
container_name = "Daily-Temp"
if not gridstore.has_container(container_name):
    container = gridstore.create_time_series_container(container_name, "Daily-Temp", "Daily-Temp", "Daily-Temp", 10000)
else:
    container = gridstore.get_container(container_name)
```

The data is then inserted into the container, ensuring efficient storage for later retrieval:

```python
# Insert data into GridDB
for index, row in data.iterrows():
    row_key = container.row_key()
    row_key.set_timestamp(row['Date'])
    row_obj = container.get_row(row_key)
    row_obj.set_field('Daily-minimum-temperatures', row['Daily minimum temperatures'])
    container.put_row(row_obj)
```

This step leverages GridDB’s capabilities for managing large-scale time-series data, making it ready for analysis.

#### Forecasting Daily Minimum Temperatures using Prophet
With the data stored in GridDB, the next step is to retrieve it for modeling. The data is fetched and prepared for the Prophet model, which requires specific column names: 'ds' for dates and 'y' for values.

```python
# Retrieve data from GridDB
rows = container.get_row_all()
data_from_griddb = pd.DataFrame([{'Date': row.get_timestamp(), 'Daily minimum temperatures': row.get_field('Daily-minimum-temperatures')} for row in rows])
data_from_griddb['Date'] = pd.to_datetime(data_from_griddb['Date'])

# Prepare data for Prophet
data_prophet = data_from_griddb.rename(columns={'Date': 'ds', 'Daily minimum temperatures': 'y'})
data_prophet = data_prophet.sort_values('ds')
```

The dataset is then split into training (70%) and testing (30%) sets to evaluate the model’s performance:

```python
from sklearn.model_selection import train_test_split

# Split data
train, test = train_test_split(data_prophet, test_size=0.3, shuffle=False)
```

The Prophet model is trained on the training data, leveraging its ability to handle seasonality and trends:

```python
from prophet import Prophet

# Initialize and train the model
model = Prophet()
model.fit(train)
```

Predictions are generated for the test period, ensuring the forecast aligns with the test data length:

```python
# Make predictions
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)
forecast = forecast[-len(test):]
```

Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), providing quantitative measures of accuracy:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE and RMSE
mae = mean_absolute_error(test['y'], forecast['yhat'])
rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
```

The results show an MAE of 2.09 and an RMSE of 2.67, indicating that, on average, predictions deviate by about 2 degrees Celsius from actual values, with slightly larger errors for outliers, suggesting reasonable accuracy for temperature forecasting.

Visualization is crucial for understanding the model’s performance. The actual and predicted values are plotted, including confidence intervals for a comprehensive view:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot actual vs predicted
plt.figure(figsize=(15,8))
plt.plot(test['ds'], test['y'], label='Actual')
plt.plot(test['ds'], forecast['yhat'], label='Predicted')
plt.fill_between(test['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Daily Minimum Temperature')
plt.title('Temperature Forecasting - Prophet Model')
plt.legend()
plt.show()
```

![Output](imgs\temp-forecast-img.png)
This visualization helps assess how well the Prophet model captures the trends and patterns in the temperature data, with the actual data (in one color) and forecast (in another) allowing for easy comparison, and confidence intervals indicating uncertainty.

#### Conclusion and Future Directions
This blog post demonstrated a complete workflow for time-series forecasting using GridDB and the Prophet model. We started by loading and cleaning the daily minimum temperature dataset, storing it in GridDB, and retrieving it for modeling. We then prepared the data, trained the Prophet model, made predictions, and evaluated the results, achieving an MAE of 2.09 and an RMSE of 2.67. The visualization provided a clear comparison between actual and predicted values, highlighting the model’s ability to capture seasonal trends.

GridDB’s scalability and Prophet’s ease of use make this approach suitable for various applications, such as weather prediction, energy consumption forecasting, and more. For further questions or assistance, readers can refer to the GridDB documentation or ask on Stack Overflow with the "griddb" tag ([Stack Overflow](https://stackoverflow.com/questions/ask?tags=griddb)). Future work could involve tuning hyperparameters, incorporating additional features, or addressing data quality issues to further improve prediction accuracy.

#### Key Citations
- [GridDB Python Client on PyPI](https://pypi.org/project/griddb-python/)
- [Facebook Prophet Official Documentation](https://facebook.github.io/prophet/)
- [Pandas Official Documentation](https://pandas.pydata.org/)
- [Matplotlib Official Documentation](https://matplotlib.org/)
- [Seaborn Official Documentation](https://seaborn.pydata.org/)
- [Scikit-learn Official Documentation](https://scikit-learn.org/)
- [Stack Overflow Questions with griddb Tag](https://stackoverflow.com/questions/ask?tags=griddb)