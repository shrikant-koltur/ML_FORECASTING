import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from category_encoders import TargetEncoder
# from sklearn.preprocessing import MinMaxScaler

company_size = pd.read_csv("./data/company_size.csv")
all_data = pd.read_csv('./data/subquery_30_04_2024.csv')

def company_size_merge(df, df_cs):
    all_data = pd.merge(df, df_cs, on='company_name', how='left')
    all_data['company_size'].fillna('Not Available', inplace=True)
    return all_data

def generate_synthetic_weather_data2(start_date, end_date):
    weather_data = pd.read_csv("./data/weather_data_daily_average.csv")
    weather_data['start_date'] = pd.to_datetime(weather_data['start_date'])

    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter weather_data for the month of the start_date
    month_filter = (weather_data['start_date'].dt.month == start_date.month)
    past_data = weather_data.loc[month_filter, :].copy()
    # print(past_data)

    # Generate new dates for the specified period
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    num_rows = len(forecast_dates)

    # Sample data from the past data for each column
    temperature_2m_degC = np.random.choice(past_data['temperature_2m_degC'], size=num_rows, replace=True)
    relative_humidity_2m_pct = np.random.choice(past_data['relative_humidity_2m_pct'], size=num_rows, replace=True)
    apparent_temperature_degC = np.random.choice(past_data['apparent_temperature_degC'], size=num_rows, replace=True)
    rainfall_mm = np.random.choice(past_data['rainfall_mm'], size=num_rows, replace=True)
    cloud_cover_pct = np.random.choice(past_data['cloud_cover_pct'], size=num_rows, replace=True)
    wind_speed_10m_kmh = np.random.choice(past_data['wind_speed_10m_kmh'], size=num_rows, replace=True)

    data = {
        'start_date': forecast_dates,
        'temperature_2m_degC': temperature_2m_degC,
        'relative_humidity_2m_pct': relative_humidity_2m_pct,
        'apparent_temperature_degC': apparent_temperature_degC,
        'rainfall_mm': rainfall_mm,
        'cloud_cover_pct': cloud_cover_pct,
        'wind_speed_10m_kmh': wind_speed_10m_kmh
    }

    synthetic_data = pd.DataFrame(data)
    return synthetic_data

def generate_synthetic_weather_data(start_date, end_date,seed=7):
    if seed is not None:
        np.random.seed(seed)

    # Define the date range for forecasting
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='2H')
    num_rows = len(forecast_dates)

    temperature_2m_degC = np.random.uniform(20, 40, num_rows)
    relative_humidity_2m_pct = np.random.uniform(40, 100, num_rows)
    apparent_temperature_degC = np.random.uniform(25, 45, num_rows)
    rainfall_mm = np.random.exponential(scale=0.5, size=num_rows) # Exponential distribution for rainfall
    cloud_cover_pct = np.random.uniform(0, 100, num_rows)
    wind_speed_10m_kmh = np.random.uniform(0, 20, num_rows)

    data = {
        'start_date': forecast_dates,
        'temperature_2m_degC': temperature_2m_degC,
        'relative_humidity_2m_pct': relative_humidity_2m_pct,
        'apparent_temperature_degC': apparent_temperature_degC,
        'rainfall_mm': rainfall_mm,
        'cloud_cover_pct': cloud_cover_pct,
        'wind_speed_10m_kmh': wind_speed_10m_kmh
    }
    synthetic_data = pd.DataFrame(data)
    return synthetic_data

def generate_dataset_by_dates(df, start_date, end_date, company_name, location_name, seed = 7):
    if seed is not None:
        np.random.seed(seed)
    # Generate random datetimes within the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    random_times = [pd.Timedelta(np.random.randint(0, 3600), unit='s') for _ in range(len(date_range))]
    start_datetimes = [date + time for date, time in zip(date_range, random_times)]
    
    # Filter data for the given company and location
    past_data = df[(df['company_name'] == company_name) & 
                   (df['location_name'] == location_name)]
    
    # Get unique job titles from past data
    unique_job_titles = past_data['job_title'].unique()

    # Calculate the desired size based on the date range
    desired_size = len(date_range)

    # Randomly select job titles from the unique list
    if len(unique_job_titles) >= desired_size:
        random_job_titles = np.random.choice(unique_job_titles, size=desired_size, replace=False)
    else:
        # If the number of unique job titles is less than the desired size,
        # duplicate job titles to match the desired size
        num_duplicates_needed = desired_size - len(unique_job_titles)
        duplicated_job_titles = np.random.choice(unique_job_titles, size=num_duplicates_needed, replace=True)
        random_job_titles = np.concatenate((unique_job_titles, duplicated_job_titles))

    # Create a DataFrame for predictions
    result = pd.DataFrame({
        'start_date': start_datetimes,
        'company_name': company_name,
        'company_id': past_data['company_id'].iloc[0],
        'location_name': location_name,
        'location_id': past_data['location_id'].iloc[0],
        'job_title': random_job_titles
    })
    return result


def create_time_feature(df):
    df['hour'] = df['start_date'].dt.hour
    df['minute'] = df['start_date'].dt.minute
    df['dayofmonth'] = df['start_date'].dt.day
    df['dayofweek'] = df['start_date'].dt.dayofweek
    df['quarter'] = df['start_date'].dt.quarter
    df['month'] = df['start_date'].dt.month
    df['year'] = df['start_date'].dt.year
    df['dayofyear'] = df['start_date'].dt.dayofyear
    return df

def preprocess_data(data, seed=7):
    """
    Preprocess the data by creating time-based features, encoding categorical features,
    and normalizing numerical features.
    """
    if seed is not None:
        np.random.seed(seed)
    # Convert date columns to datetime format
    data['start_date'] = pd.to_datetime(data['start_date'])
    start_date_min = data['start_date'].min()
    end_date_max = data['start_date'].max()
    synthetic_data = generate_synthetic_weather_data2(start_date_min, end_date_max)
    merged_df = pd.merge(data, synthetic_data, on='start_date', how='left')
    merged_df[['temperature_2m_degC', 'relative_humidity_2m_pct', 'apparent_temperature_degC']] = merged_df[['temperature_2m_degC', 'relative_humidity_2m_pct', 'apparent_temperature_degC']].astype('float64')
    
    # Specify numeric columns
    numeric_columns = ['temperature_2m_degC', 'relative_humidity_2m_pct', 'apparent_temperature_degC','rainfall_mm', 'cloud_cover_pct', 'wind_speed_10m_kmh']

    # Fill NaN values with the mean of numeric columns
    df_filled = merged_df.copy()
    df_filled[numeric_columns] = df_filled[numeric_columns].fillna(df_filled[numeric_columns].mean())
    all_data = company_size_merge(df_filled, company_size)
    all_data = create_time_feature(all_data)
    hour_filter = (all_data['hour'] == 0)
    all_data.loc[hour_filter, 'hour'] = np.random.uniform(2, 12, size=all_data[hour_filter].shape[0]).round().astype('float64')
    # all_data.rename(columns={"total_people_hired":"people_hired"}, inplace=True)
    
    target_encoder = joblib.load("./models/weather_sampling_split_target_encoder_03_05.pkl")

    all_data = all_data.drop(["start_date"], axis=1)
    all_data = target_encoder.transform(all_data)
    
    # Normalize numerical features
    numerical_cols = ['company_id', 'location_id', 'temperature_2m_degC', 'relative_humidity_2m_pct',
       'apparent_temperature_degC', 'rainfall_mm', 'cloud_cover_pct',
       'wind_speed_10m_kmh', 'company_size', 'hour', 'minute', 'dayofmonth',
       'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    
    scaler = joblib.load('./models/weather_sampling_split_scaler_03_05.pkl')
    all_data[numerical_cols] = scaler.transform(all_data[numerical_cols])
    X_train = joblib.load("./models/weather_sampling_split_X_train_03_05.pkl")
    
    # Check if the input DataFrame has the same columns as the training data
    expected_columns = X_train.columns
    missing_cols = set(expected_columns) - set(all_data.columns)
    if missing_cols:
        
        # Add missing columns with default value 0
        for col in missing_cols:
            all_data[col] = 0
    input_df = all_data[expected_columns]
    return input_df

def predict_people_hired(data):
    """
    Predict the number of people hired given a DataFrame containing the input features.
    """
    # Load the trained model, target encoder, and scaler
    model = joblib.load('./models/weather_sampling_split_model_03_05.pkl')

    # Preprocess the data
    data = preprocess_data(data)

    # Make the prediction
    predictions = model.predict(data)

    return predictions

st.title("People Hiring Prediction")

# Date range selection
start_date = st.date_input("Select Start Date", value=datetime(2024, 5, 1), min_value=datetime(2024, 5, 1), max_value=datetime(2025, 5, 31))
end_date = st.date_input("Select End Date", value=datetime(2024, 5, 31), min_value=datetime(2024, 5, 1), max_value=datetime(2025, 5, 31))

# Convert the selected dates to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Company and location selection
company_list = ['All Companies'] + list(all_data['company_name'].unique())
selected_company = st.selectbox("Select a Company", company_list)

if selected_company != 'All Companies':
    location_list = ['All Locations'] + list(all_data[all_data['company_name'] == selected_company]['location_name'].unique())
    selected_location = st.selectbox("Select a Location", location_list)
else:
    selected_location = 'All Locations'

# Generate dataset and predictions
if selected_company != 'All Companies' and selected_location != 'All Locations':
    past_data = all_data[(all_data['company_name'] == selected_company) & (all_data['location_name'] == selected_location)]
    new_data = generate_dataset_by_dates(all_data, start_date_str, end_date_str, selected_company, selected_location)
    predictions = predict_people_hired(new_data)

    # Round the predictions based on the conditions
    rounded_predictions = np.zeros_like(predictions)

    for i in range(len(predictions)):
        if predictions[i] < 0.5:
            rounded_predictions[i] = np.floor(predictions[i])
        else:
            rounded_predictions[i] = np.ceil(predictions[i])

        # Make negative values zero
        if rounded_predictions[i] < 0:
            rounded_predictions[i] = 0

    # Convert rounded predictions to integer type
    rounded_predictions = rounded_predictions.astype(int)

    pred_data = pd.DataFrame(rounded_predictions, columns=['Predictions'])

    full_data = pd.concat([new_data, pred_data], axis=1)
    # print(full_data.columns)
    # print(full_data['job_title'].value_counts())
    # print(full_data['Predictions'].value_counts())
    # print(full_data['Predictions'].sum())
    # print(full_data.info())
    # print(new_data.info())
    # print(pred_data.info())

    # st.dataframe(past_data)

    # Convert 'start_date' to datetime format
    # past_data['start_date'] = pd.to_datetime(past_data['start_date'])

    # Group by 'start_date' and sum the 'total_people_hired' column
    sum_by_date = past_data.groupby('start_date')['total_people_hired'].sum().reset_index()
    fig = go.Figure()

    if 'total_people_hired' in past_data.columns:
        sum_predictions_by_date = past_data.groupby('start_date')['total_people_hired'].sum().reset_index()
        fig.add_trace(go.Scatter(x=sum_predictions_by_date['start_date'], y=sum_predictions_by_date['total_people_hired'], mode='markers', name='total_people_hired'))
        fig.add_trace(go.Scatter(x=sum_predictions_by_date['start_date'], y=sum_predictions_by_date['total_people_hired'], mode='lines', name='Trend Line'))

    fig.update_layout(title='Past Total People Hired Over Time', xaxis_title='Date', yaxis_title='Total People Hired', xaxis=dict(tickangle=-45), legend=dict(x=0, y=1.0), template='plotly')
    
    # Show the plot
    st.subheader("Past Total People Hired")
    st.plotly_chart(fig)
    # Group by date and sum the 'Predictions' column (location-wise)
    sum_predictions_by_date_location = full_data.groupby('start_date')['Predictions'].sum().reset_index()

    # Create a trace for the data points
    trace_data = go.Scatter(x=sum_predictions_by_date_location['start_date'], y=sum_predictions_by_date_location['Predictions'], mode='markers', name='Predicted')

    # Create a trace for the trend line
    trend_line = go.Scatter(x=sum_predictions_by_date_location['start_date'], y=sum_predictions_by_date_location['Predictions'], mode='lines', name='Trend Line')

    # Create the figure and add the traces
    fig_location = go.Figure()
    fig_location.add_trace(trace_data)
    fig_location.add_trace(trend_line)

    # Sum of predictions
    sum_predictions = full_data['Predictions'].sum()
    st.markdown("### Total Predictions")
    st.write(f"<p style='color:green; font-size:20px;'>{sum_predictions}</p>", unsafe_allow_html=True)

    # Customize layout for location-wise plot
    if selected_location == 'All Locations':
        title_location = 'Predictions Over Time (All Locations)'
    else:
        title_location = f'Predictions Over Time for {selected_location} from {start_date_str} - {end_date_str}'

    fig_location.update_layout(title=title_location, xaxis_title='Date', yaxis_title='Predictions',
                                xaxis=dict(tickangle=-45), legend=dict(x=0, y=1.0, orientation='h'),
                                height=500, width=1000, template='seaborn')

    # Show the location-wise plot
    st.subheader("Location Plot")
    st.plotly_chart(fig_location)

    # Pie chart for job title distribution
    job_title_counts = full_data['job_title'].value_counts()

    fig_pie = go.Figure(data=[go.Pie(labels=job_title_counts.index, values=job_title_counts.values, hole=0.5)])
    fig_pie.update_layout(title='Job Title Distribution', height=600, width=800)
    st.subheader("Job Title Distribution")
    st.plotly_chart(fig_pie)

    # Pie chart for prediction distribution
    prediction_counts = full_data['Predictions'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=prediction_counts.index, values=prediction_counts.values, hole=0.5)])
    fig_pie.update_layout(title='Prediction Distribution', height=600, width=800)
    st.subheader("Prediction Distribution")
    st.plotly_chart(fig_pie)

    # Bar chart for company-wise predictions
    location_predictions = full_data.groupby('location_name')['Predictions'].sum().reset_index()
    fig_bar = go.Figure(data=[go.Bar(x=location_predictions['location_name'], y=location_predictions['Predictions'])])
    fig_bar.update_layout(title='location-wise Predictions', xaxis_title='location', yaxis_title='Predictions', height=600, width=800)
    st.subheader("location-wise Predictions")
    st.plotly_chart(fig_bar)

    # Violin Plot
    fig_violin = go.Figure()
    for company in full_data['company_name'].unique():
        company_data = full_data[full_data['company_name'] == company]['Predictions']
        fig_violin.add_trace(go.Violin(x=[company] * len(company_data), y=company_data, name=company, points='all', jitter=0.8, scalemode='count'))
    fig_violin.update_layout(title='Violin Plot of Predictions by Company', xaxis_title='Company', yaxis_title='Predictions',
                              height=600, width=800, template='seaborn')
    st.subheader("Violin Plot of Predictions by Company")
    st.plotly_chart(fig_violin)
else:
    st.warning("Please select a company and location to generate predictions.")

# st.title("People Hiring Prediction")
# from datetime import datetime

# # Date range selection
# start_date = st.date_input("Select Start Date", value=datetime(2024, 5, 1), min_value=datetime(2024, 5, 1), max_value=datetime(2025, 5, 31))
# end_date = st.date_input("Select End Date", value=datetime(2024, 5, 31), min_value=datetime(2024, 5, 1), max_value=datetime(2025, 5, 31))

# # Convert the selected dates to strings
# start_date_str = start_date.strftime('%Y-%m-%d')
# end_date_str = end_date.strftime('%Y-%m-%d')

# # Company and location selection
# company_list = ['All Companies'] + list(all_data['company_name'].unique())
# selected_company = st.selectbox("Select a Company", company_list)

# if selected_company != 'All Companies':
#     location_list = ['All Locations'] + list(all_data[all_data['company_name'] == selected_company]['location_name'].unique())
#     selected_location = st.selectbox("Select a Location", location_list)
# else:
#     selected_location = 'All Locations'

# # Generate dataset and predictions
# if selected_company != 'All Companies' and selected_location != 'All Locations':
#     new_data = generate_dataset_by_dates(all_data, start_date_str, end_date_str, selected_company, selected_location)
#     predictions = predict_people_hired(new_data)

#     # Round the predictions based on the conditions
#     rounded_predictions = np.zeros_like(predictions)

#     for i in range(len(predictions)):
#         if predictions[i] < 0.5:
#             rounded_predictions[i] = np.floor(predictions[i])
#         else:
#             rounded_predictions[i] = np.ceil(predictions[i])

#         # Make negative values zero
#         if rounded_predictions[i] < 0:
#             rounded_predictions[i] = 0

#     # Convert rounded predictions to integer type
#     rounded_predictions = rounded_predictions.astype(int)

#     pred_data = pd.DataFrame(rounded_predictions, columns=['Predictions'])

#     full_data = pd.concat([new_data, pred_data], axis=1)

#     # Group by date and sum the 'Predictions' column (location-wise)
#     sum_predictions_by_date_location = full_data.groupby('start_date')['Predictions'].sum().reset_index()
    
#     # Create a trace for the data points
#     trace_data = go.Scatter(x=sum_predictions_by_date_location['start_date'],
#                             y=sum_predictions_by_date_location['Predictions'],
#                             mode='markers', name='Predicted')

#     # Create a trace for the trend line
#     trend_line = go.Scatter(x=sum_predictions_by_date_location['start_date'],
#                             y=sum_predictions_by_date_location['Predictions'],
#                             mode='lines', name='Trend Line')

#     # Create the figure and add the traces
#     fig_location = go.Figure()
#     fig_location.add_trace(trace_data)
#     fig_location.add_trace(trend_line)
    
#     # Sum of preds
#     sum_predictions = full_data['Predictions'].sum()
#     st.markdown("### Total Predictions")
#     st.write(f"<p style='color:green; font-size:20px;'>{sum_predictions}</p>", unsafe_allow_html=True)
    
#     # Customize layout for location-wise plot
#     if selected_location == 'All Locations':
#         title_location = 'Predictions Over Time (All Locations)'
#     else:
#         title_location = f'Predictions Over Time for {selected_location}'

#     fig_location.update_layout(title=title_location, xaxis_title='Date', yaxis_title='Predictions',
#                             xaxis=dict(tickangle=-45), legend=dict(x=0, y=1.0, orientation='h'),
#                             height=500, width=1000, template='seaborn')

#     # Show the location-wise plot
#     st.subheader("Location Plot")
#     st.plotly_chart(fig_location)