# Time-Series Forecasting with GridDB and Meta's Prophet Model

![Output](https://github.com/FastianAbdullah/Time-Classification-with-GridDb-and-Metas-Prophet/blob/main/imgs/temp-forecast-img.png)

This article shows how to build a time series forecasting model for daily minimum temperature using GridDB and Prophet Model.We want to predict future daily minimum temperatures by learning from past weather data.Think of it like this - if you know how cold it's been every day for the past few years, you can make educated guesses about how cold it will be in the coming days or months.

We will retrieve historical daily minimum temperatures from the kaggle dataset, insert it into a GridDB time series container and use that data to train a forecasting model which is meta prophet model developed by facebook, a specialized additive model where non-linear trends are fit with yearly, weekly and daily seasonality.

GridDB is a robust NOSQL database optimized for efficiently handling large volumes of real-time data. Its advanced in-memory processing and time series data management make it ideal for big data and IoT applications.

## Prerequisites

You need to install the following libraries to run codes in this article.

1. GridDB C Client
2. GridDB Python client

Instructions for installing these clients are available on [GridDB Python Package Index (Pypi)](https://pypi.org/project/griddb-python/).

You must also install Prophet, Numpy, Pandas, Seaborn, Scikit-Learn and Matplotlib libraries.

The scripts below will help you install and import the necessary libraries for running codes.

```bash
%pip install prophet seaborn numpy pandas scikit-learn matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import griddb_python as griddb
from prophet import Prophet
```

## Inserting Time Series Data Into GridDB

The first step is to insert the time series data we want to forecast into GridDB. This section explain the following:

### Downloading and Importing Daily Minimum Temperature Data from Kaggle

We will perform forecasting using the [Daily Minimum Temperature dataset from Kaggle](https://www.kaggle.com/datasets/suprematism/daily-minimum-temperatures)

The following line will import the dataset and show the first five entries in it.

```python
data = pd.read_csv('daily-minimum-temperatures.csv')
print(data.head(5))
```

Output:

|   | Date     | Daily minimum temperatures |
|---|----------|----------------------------|
| 0 | 1/1/1981 | 20.7                       |
| 1 | 1/2/1981 | 17.9                       |
| 2 | 1/3/1981 | 18.8                       |
| 3 | 1/4/1981 | 14.6                       |
| 4 | 1/5/1981 | 15.8                       |

The Dataset contains each day's minimum temperature starting from 1st January 1981 till 31st December 1990.

Now before moving forward check the data types of columns.

```python
# Check the data info
data.info()
```

Output:

``` markdown
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3650 entries, 0 to 3649
Data columns (total 2 columns):
 #   Column                      Non-Null Count  Dtype 
---  ------                      --------------  ----- 
 0   Date                        3650 non-null   object
 1   Daily minimum temperatures  3650 non-null   object
dtypes: object(2)
memory usage: 57.2+ KB
```

It shows that both columns are of object type. We need to convert the date column into pandas datatime data type and daily minimum temperature to float type.

```python
# Pick only the floating point number as it is an object right now from Min Temperature to convert it into numeric.
data['Daily minimum temperatures'] = data['Daily minimum temperatures'].str.extract('(\d+\.?\d*)')[0]

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Convert temperature column to numeric
data['Daily minimum temperatures'] = pd.to_numeric(data['Daily minimum temperatures'])

# Check the info again.
data.info()
```

Output:

```markdown
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3650 entries, 0 to 3649
Data columns (total 2 columns):
 #   Column                      Non-Null Count  Dtype         
---  ------                      --------------  -----         
 0   Date                        3650 non-null   datetime64[ns]
 1   Daily minimum temperatures  3650 non-null   float64       
dtypes: datetime64[ns](1), float64(1)
memory usage: 57.2 KB
```

Now, we can plot only a single year data from our dataset to see the trend of Daily Minimum Temperature which first decreases and then increases over the course of a year.

```python
# plot the trend of temperatures over time using lineplot.
def plot_trend(data) -> None:
    plt.figure(figsize=(15,8))
    sns.lineplot(data=data,x='Date',y='Daily minimum temperatures')
    plt.xlabel('Date')
    plt.ylabel('Daily Min Temperature')
    plt.title('Daily Min Temperature Over Time')
    plt.show()

    return None

# Check the trend of temperature over a single year.
plot_trend(data[data['Date'].dt.year == 1981])
```

Output:

![Temperature-over-a-Single-Year](https://github.com/FastianAbdullah/Time-Classification-with-GridDb-and-Metas-Prophet/blob/main/imgs/temp-over-a-single-year.png)

### Connect to GridDB

To connect to GridDB, you need to create an object of the `StoreFactory` class. Next, call the `get_store()` method on the store factory object and pass the DB host and cluster name, user, and password.

To test if the connection is successful, call the `get_container()` method and pass it the name of any container. If you see the following output, your connection is successful.

```python
# GridDB connection details
DB_HOST = "127.0.0.1:10001"
DB_CLUSTER = "myCluster"
DB_USER = "admin"
DB_PASS = "admin"

# creating a connection

factory = griddb.StoreFactory.get_instance()

try:
    gridstore = factory.get_store(
        notification_member = DB_HOST,
        cluster_name = DB_CLUSTER,
        username = DB_USER,
        password = DB_PASS
    )

    container1 = gridstore.get_container("Daily-Temp")
    if container1 == None:
        print("Container does not exist")
    print("Successfully connected to GridDB")

except griddb.GSException as e:
    for i in range(e.get_error_stack_size()):
        print("[", i, "]")
        print(e.get_error_code(i))
        print(e.get_location(i))
        print(e.get_message(i))
```

Output:

``` markdown
Container does not exist
Successfully connected to GridDB
```

### Create Container

Now, our connection to the GridDB server is successful. Let's move on to create a container for our dataset which will help us to insert our data.

GridDB containers are created by specifying the `ContainerInfo` class objects which consist of `ContainerName` , `Columns info list` and `Container Type`.

The `Container Name` could be any name you like.However, the `Columns info list` must be a list of lists, each nested list containing the column name and the column type. Lastly, for `Container Type` specify the type of container which in our case is `griddb.ContainerType.TIME_SERIES`

Finally, call the `put_container()` method and pass to it the ContainerInfo class object to create a container in the GridDB.

```python
try:
    # Create a new Collection of Store.
    coninfo = griddb.ContainerInfo("Daily-Temp",
                                   [
                                       ["Date", griddb.Type.TIMESTAMP],
                                       ["Daily-minimum-temperatures", griddb.Type.FLOAT]
                                   ],
                                   type=griddb.ContainerType.TIME_SERIES)
    gridstore.put_container(coninfo)
    container = gridstore.get_container("Daily-Temp")
    if cont == None:
        print("Failed to create container")
    else:
        print(f"Container Created Successfully")
except griddb.GSException as e:
   for i in range(e.get_error_stack_size()):
       print("[", i, "]")
       print(e.get_error_code(i))
       print(e.get_location(i))
       print(e.get_message(i))
```

Output:

``` markdown
Container Created Successfully
```

Now within our `container` variable we have successfully got the container. We will use it next to insert the data into it.

### Insert Daily Minimum Temperature Data into GridDB

To insert the data into our `container` object, we will iterate through all the rows of the data and by using the `put` method we will place the `Date` Column and `Daily Minimum Temperature` column.

```python
# Put Data into Grid DataBase.
try:
    for index,row in data.iterrows():
        cont.put([row["Date"],row["Daily minimum temperatures"]])
    print(f"Insertion Completed Successfully.")
except griddb.GSException as e:
   for i in range(e.get_error_stack_size()):
       print("[", i, "]")
       print(e.get_error_code(i))
       print(e.get_location(i))
       print(e.get_message(i)) 
```

Output

``` markdown
Insertion Completed Successfully.
```

## Forecasting Daily Minimum Temperatures using Meta Prophet Model

With the data stored in GridDB, the next step is to retrieve it for modeling. To do so, you can use the `get_container()` method to get the created container by passing the container name you want to retrieve.

Call the `SELECT *` query using the container's `query()` method. Next, call the `fetch` method to get the dataset object. Finally, call the `fetch_rows()` method to store the dataset into a pandas DataFrame.

```python
# Retrieve Data from Grid DB for Model.
try:
    temperature_container = gridstore.get_container("Daily-Temp")
    query = temperature_container.query("select *")
    rs = query.fetch()
    data=rs.fetch_rows()
    print(f"Successfully retrieved {len(data)} rows from GridDB.")
except griddb.GSException as e:
   for i in range(e.get_error_stack_size()):
       print("[", i, "]")
       print(e.get_error_code(i))
       print(e.get_location(i))
       print(e.get_message(i))

data.head(5)
```

Output:

|   | Date       | Daily-minimum-temperatures |
|---|------------|----------------------------|
| 0 | 1981-01-01 | 20.700001                  |
| 1 | 1981-01-02 | 17.900000                  |
| 2 | 1981-01-03 | 18.799999                  |
| 3 | 1981-01-04 | 14.600000                  |
| 4 | 1981-01-05 | 15.800000                  |

### Preparing Data and Train the Model

The input to Prophet is always a dataframe with two columns: ds and y . The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast. So let's update our dataset accordingly.

```python
# Prepare Data Format for Prophet model.
prophet_df = pd.DataFrame(data={
    'ds' : data['Date'],
    'y': data['Daily-minimum-temperatures']
})

# Update its pre-generated index column and remove it.
prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
prophet_df.head(5)
```

Output:

|   | ds         | y         |
|---|------------|-----------|
| 0 | 1981-01-01 | 20.700001 |
| 1 | 1981-01-02 | 17.900000 |
| 2 | 1981-01-03 | 18.799999 |
| 3 | 1981-01-04 | 14.600000 |
| 4 | 1981-01-05 | 15.800000 |

The dataset is then split into training (70%) and testing (30%) sets to evaluate the model’s performance:

```python
from sklearn.model_selection import train_test_split

# Split data
train_data, test_data = train_test_split(prophet_df, test_size=0.3, shuffle=False)
print("Training set:")
print(train_data.shape)
print("\nTest Set:")
print(test_data.shape)
```

Output:

``` markdown
Training set:
(2555, 2)

Test Set:
(1095, 2)
```

Next, we will use the Prophet Model and train the model on `train_data`.

```python
# Train the Model on Train set.

# Trend Checks.
DAILY_CHECK=True
WEEKLY_CHECK=False
YEARLY_CHECK=True

model = Prophet(
    seasonality_mode='additive',
    daily_seasonality=DAILY_CHECK,
    weekly_seasonality=WEEKLY_CHECK,
    yearly_seasonality=YEARLY_CHECK,
    changepoint_prior_scale=0.05,  # Controls flexibility of trend
    seasonality_prior_scale=10     # Controls strength of seasonality
)

model.fit(train_data)
```

### Make Predictions and Evaluate Model Performance

Predictions are generated for the test period, ensuring the forecast aligns with the test data length.

```python
# Make predictions
def make_predictions(model,periods,freq='D'):
    future = model.make_future_dataframe(periods=periods,freq='D')
    forecast = model.predict(future)
    print(f"Length of ForeCast: {len(forecast)}")
    test_forecast = forecast.tail(periods).copy()

    median = test_forecast['yhat'].values           # Main prediction
    lower = test_forecast['yhat_lower'].values      # Lower bound
    upper = test_forecast['yhat_upper'].values      # Upper bound
    
    return lower, median, upper, test_forecast

# Make Predictions.
lower_bound, pred , upper_bound , forecast = make_predictions(model,periods=len(test_data))
```

Now, let's evaluate the predictions.Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), providing quantitative measures of accuracy:

```python
def evaluate_model(pred,ground_truth,model_name="Prophet"):
    # Evaluate predictions with Ground truth.
    mae = mean_absolute_error(ground_truth, pred)
    rmse = np.sqrt(mean_squared_error(ground_truth, pred))

    print(f"\n{model_name} Model Performance:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    return mae, rmse

# Evaluate Model.
evaluate_model(pred=pred,ground_truth=test_data['y'].values)
```

Output:

``` markdown
Prophet Model Performance:
MAE:  2.09
RMSE: 2.67
```

The results show an MAE of 2.09 and an RMSE of 2.67, indicating that, on average, predictions deviate by about 2 degrees Celsius from actual values, with slightly larger errors for outliers, suggesting reasonable accuracy for temperature forecasting.

### Visualizations

Finally, let's plot some visualizations to get a better understanding of model overall performance.Let's plot training set, test set and the predictions along with 80 percent confidence interval.

```python
#Plot Time Series Graph Showing Train Part, Test Part and Predicted Part.
def complete_plot_timeseries(pred, test_data, train_data=None, lower_bound=None, upper_bound=None):
    fig = plt.figure(facecolor='w', figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    # Plot training data as black dots
    if train_data is not None:
        ax.plot(train_data['ds'].values, train_data['y'], 'k.', alpha=0.5, label="Training Data")
    
    # Plot test data as green dots
    ax.plot(test_data['ds'].values, test_data['y'], 'go', alpha=0.7, label="Actual Test")
    
    # Plot predictions as blue line
    ax.plot(test_data['ds'].values, pred, ls='-', c='#0072B2', linewidth=2, label="Forecast")
    
    # Fill confidence interval
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(test_data['ds'].values, lower_bound, upper_bound, 
                       color='#0072B2', alpha=0.2, label="80% Confidence Interval")
    
    # Add vertical line to show train/test split
    if train_data is not None:
        cutoff_date = test_data['ds'].iloc[0]
        ax.axvline(x=cutoff_date, color='gray', lw=3, alpha=0.6, linestyle='--')
        
        # Add text annotations
        ax.text(x=train_data['ds'].iloc[len(train_data)//2], y=test_data['y'].max()*0.9, 
               s='Training', color='black', fontsize=14, fontweight='bold', alpha=0.8)
        ax.text(x=test_data['ds'].iloc[len(test_data)//2], y=test_data['y'].max()*0.9, 
               s='Test & Forecast', color='black', fontsize=14, fontweight='bold', alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Forecasting - Prophet Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return None

complete_plot_timeseries(pred=pred,test_data=test_data,train_data=train_data,lower_bound=lower_bound,upper_bound=upper_bound)
```

Output:
![Temp-forecast-img](https://github.com/FastianAbdullah/Time-Classification-with-GridDb-and-Metas-Prophet/blob/main/imgs/temp-forecast-img.png)

The above output shows that our model performs well and can capture the trends in the training dataset. The predictions are close to the values in the test set.

We can also take out the training part and only plot the test set with predictions and confidence intervals for better and clear analysis.

```python
#Plot Time Series Graph Showing Test Part and Predicted Part.
def test_data_plot_timeseries(pred, test_data, lower_bound=None, upper_bound=None):
    fig = plt.figure(facecolor='w', figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    # Plot test data as green dots
    ax.plot(test_data['ds'].values, test_data['y'], 'go', alpha=0.7, label="Actual Test")
    
    # Plot predictions as blue line
    ax.plot(test_data['ds'].values, pred, ls='-', c='#0072B2', linewidth=2, label="Forecast")
    
    # Fill confidence interval
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(test_data['ds'].values, lower_bound, upper_bound, 
                       color='#0072B2', alpha=0.2, label="80% Confidence Interval")
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Forecasting - Prophet Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return None

test_data_plot_timeseries(pred,test_data=test_data,lower_bound=lower_bound,upper_bound=upper_bound)
```

Output:
![Test-Data-Img](https://github.com/FastianAbdullah/Time-Classification-with-GridDb-and-Metas-Prophet/blob/main/imgs/test-data-forecast.png)

The above result shows that the majority of data points are covered by our prediction interval which is a sign of great model performance on our time-series dataset.

## Conclusion

This blog post demonstrated a complete workflow for time-series forecasting using GridDB and the Prophet model. We started by loading and cleaning the daily minimum temperature dataset, storing it in GridDB, and retrieving it for modeling. We then prepared the data, trained the Prophet model, made predictions, and evaluated the results, achieving an MAE of 2.09 and an RMSE of 2.67. The visualization provided a clear comparison between actual and predicted values, highlighting the model’s ability to capture seasonal trends.

> If you have any questions about the blog, please create a Stack Overflow post here [https://stackoverflow.com/questions/ask?tags=griddb](https://stackoverflow.com/questions/ask?tags=griddb).
> Make sure that you use the "griddb" tag so our engineers can quickly reply to your questions.
