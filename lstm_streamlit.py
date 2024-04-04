import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import random
import torch
from tqdm.auto import tqdm
import warnings
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
import time
import matplotlib.dates as mdates
import geopandas as gpd
import folium
import base64
import calendar

# Ignoring warnings
warnings.filterwarnings('ignore')

# Setting the device for PyTorch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {DEVICE}')

# Streamlit app title and description
st.title('Hydrological Model App 💦')
st.write('This app allows you to analyze and model hydrological data using LSTM and Transformer models.')

# Loading and concatenating CSV files into a DataFrame
@st.cache_data
def load_default_data():
    data_dir = '/Users/doudou.ba/Downloads/dbstream'
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    basin_dataframes = [pd.read_csv(file, parse_dates=['Date'], index_col='Date') for file in csv_files]
    data = pd.concat(basin_dataframes, axis=0)
    return data

default_data = load_default_data()

# Data upload
st.sidebar.subheader('Data Upload')
uploaded_file = st.sidebar.file_uploader('Upload CSV', type='csv')

if uploaded_file is not None:
    # Load user-uploaded data
    uploaded_data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    
    # Check if required columns exist in the uploaded data
    required_columns = ['P', 'T', 'Q']
    missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
    
    if missing_columns:
        st.warning(f"The following required columns are missing in the uploaded data: {', '.join(missing_columns)}. Please ensure the uploaded data contains these columns.")
    else:
        # Replace the default data with the user-uploaded data
        data = uploaded_data
        st.success('Data uploaded successfully!')
else:
    data = default_data

# Check if the data is empty
if data.empty:
    st.warning("No data available. Please upload a valid CSV file.")
    st.stop()

# Replace negative values with NaN
data.loc[data['P'] < 0, 'P'] = np.nan
data.loc[data['Q'] < 0, 'Q'] = np.nan

# Fill NaN values with the mean
data['P'] = data['P'].fillna(data['P'].mean())
data['Q'] = data['Q'].fillna(data['Q'].mean())

# Randomly selecting basins for analysis
basins = data['Basin'].unique()

# User options for basin selection
basin_selection = st.sidebar.radio('Basin Selection', ('Random', 'Select Basins', 'All Basins'))

if basin_selection == 'Random':
    num_basins = st.sidebar.number_input('Number of Random Basins', min_value=1, max_value=len(basins), value=5, step=1)
    selected_basins = random.sample(list(basins), num_basins)
elif basin_selection == 'Select Basins':
    selected_basins = st.sidebar.multiselect('Select Basins', basins)
else:
    selected_basins = list(basins)

# Filter the data based on selected basins
if basin_selection != 'All Basins':
    data = data[data['Basin'].isin(selected_basins)]

# Check if the filtered data is empty
if data.empty:
    st.warning("No data found for the selected basins. Please select different basins.")
    st.stop()

# Hyperparameters
st.sidebar.subheader('Hyperparameters')
epoch_num = st.sidebar.number_input('Number of Epochs', value=50, min_value=1, step=1)
optimizer = st.sidebar.selectbox('Optimizer', ('adam', 'sgd', 'rmsprop'))
batch_size = st.sidebar.number_input('Batch Size', value=32, min_value=1, step=1)

# Model selection
st.sidebar.subheader('Model Selection')
train_lstm = st.sidebar.checkbox('Train LSTM Model', value=True)
train_transformer = st.sidebar.checkbox('Train Transformer Model', value=True)

# Train button and stop button
train_button = st.sidebar.button('Train Model')
stop_button = st.sidebar.button('Stop Training')

# Data download
def download_data(data, file_name):
    csv = data.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Download selected data
st.sidebar.subheader('Download Data')
file_name = st.sidebar.text_input('File Name', 'selected_data.csv')
st.sidebar.markdown(download_data(data, file_name), unsafe_allow_html=True)

# Define the functions for performance metrics
def calculate_nse(observed, simulated):
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def calculate_kge(observed, simulated):
    observed = np.reshape(observed, (-1,))
    simulated = np.reshape(simulated, (-1,))
    correlation = np.corrcoef(observed, simulated)[0, 1]
    std_observed, std_simulated = np.std(observed), np.std(simulated)
    mean_observed, mean_simulated = np.mean(observed), np.mean(simulated)
    kge = 1 - np.sqrt((correlation - 1)**2 + (std_simulated / std_observed - 1)**2 + (mean_simulated / mean_observed - 1)**2)
    return kge

def calculate_mse(observed, simulated):
    return mean_squared_error(observed, simulated)

def calculate_rmse(observed, simulated):
    return np.sqrt(mean_squared_error(observed, simulated))

def plot_results(basin_data, dates, observed, lstm_pred, transformer_pred):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    tmin = basin_data.loc[dates, 'T'].values
    rain = basin_data.loc[dates, 'P'].values
    snow_days = tmin < 0
    print(f"Total days with temperatures below zero (potential snow days): {np.sum(snow_days)}")
    adjusted_snow = np.where(snow_days, rain, 0)
    adjusted_rain = np.where(~snow_days, rain, 0)
    ax1.bar(dates[snow_days], -adjusted_snow[snow_days], width=1, color='cyan', label='Snow (mm/day)')
    ax1.bar(dates[~snow_days], -adjusted_rain[~snow_days], width=1, color='black', label='Rain (mm/day)')
    ax1.set_ylabel('Precipitation (mm/day) [Inverted]')
    ax1.set_xlabel('Date')
    ax1.invert_yaxis()
    ax1.set_ylim(-max(rain.max(), adjusted_snow.max()) * 1.2, 0)
    ax2 = ax1.twinx()
    ax2.plot(dates, observed, 'b-', label='Observed Discharge')
    if train_lstm:
        ax2.plot(dates, lstm_pred, 'orange', label='LSTM Predictions')
    if train_transformer:
        ax2.plot(dates, transformer_pred, 'green', label='Transformer Predictions')
    ax2.set_ylabel('Discharge')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.xticks(rotation=45)
    plt.title('Hydrological Model Output with Precipitation and Discharge')
    st.pyplot(fig)

def train_and_evaluate(basin_data, selected_basin):
    scaler = MinMaxScaler()
    basin_data_scaled = scaler.fit_transform(basin_data[['P', 'T', 'Q']])
    basin_data_scaled = pd.DataFrame(basin_data_scaled, columns=['P', 'T', 'Q'], index=basin_data.index)
    seq_length = 7
    X, y = [], []
    for i in range(seq_length, len(basin_data_scaled)):
        X.append(basin_data_scaled.iloc[i - seq_length:i][['P', 'T']].values)
        y.append(basin_data_scaled.iloc[i]['Q'])
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_evaluation_metrics = {}
    test_evaluation_metrics = {}

    if train_lstm:
        lstm_model = Sequential([LSTM(64, activation='relu', input_shape=(seq_length, 2)), Dense(1)])
        lstm_model.compile(optimizer=optimizer, loss='mse')
        history = lstm_model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_size, verbose=0)

        # Evaluate on the training set
        y_train_pred = lstm_model.predict(X_train)
        train_mse = calculate_mse(y_train, y_train_pred)
        train_rmse = calculate_rmse(y_train, y_train_pred)
        train_nse = calculate_nse(y_train, y_train_pred)
        train_kge = calculate_kge(y_train, y_train_pred)
        train_evaluation_metrics['LSTM'] = {'MSE': train_mse, 'RMSE': train_rmse, 'NSE': train_nse, 'KGE': train_kge}

        # Evaluate on the test set
        y_test_pred = lstm_model.predict(X_test)
        test_mse = calculate_mse(y_test, y_test_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        test_nse = calculate_nse(y_test, y_test_pred)
        test_kge = calculate_kge(y_test, y_test_pred)
        test_evaluation_metrics['LSTM'] = {'MSE': test_mse, 'RMSE': test_rmse, 'NSE': test_nse, 'KGE': test_kge}

        # Check if the stop button is clicked
        if stop_button:
            st.write(f"Training stopped for LSTM model.")

    if train_transformer:
        transformer_input = Input(shape=(seq_length, 2))
        transformer_x = LayerNormalization(epsilon=1e-6)(transformer_input)
        transformer_x = MultiHeadAttention(num_heads=2, key_dim=seq_length)(transformer_x, transformer_x, transformer_x)
        transformer_x = Dropout(0.1)(transformer_x)
        transformer_x = LayerNormalization(epsilon=1e-6)(transformer_x)
        transformer_x = GlobalAveragePooling1D()(transformer_x)
        transformer_x = Dense(64, activation='relu')(transformer_x)
        transformer_x = Dropout(0.1)(transformer_x)
        transformer_output = Dense(1)(transformer_x)
        transformer_model = Model(inputs=transformer_input, outputs=transformer_output)
        transformer_model.compile(optimizer=optimizer, loss='mse')
        history = transformer_model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_size, verbose=0)

        # Evaluate on the training set
        y_train_pred = transformer_model.predict(X_train)
        train_mse = calculate_mse(y_train, y_train_pred)
        train_rmse = calculate_rmse(y_train, y_train_pred)
        train_nse = calculate_nse(y_train, y_train_pred)
        train_kge = calculate_kge(y_train, y_train_pred)
        train_evaluation_metrics['Transformer'] = {'MSE': train_mse, 'RMSE': train_rmse, 'NSE': train_nse, 'KGE': train_kge}

        # Evaluate on the test set
        y_test_pred = transformer_model.predict(X_test)
        test_mse = calculate_mse(y_test, y_test_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        test_nse = calculate_nse(y_test, y_test_pred)
        test_kge = calculate_kge(y_test, y_test_pred)
        test_evaluation_metrics['Transformer'] = {'MSE': test_mse, 'RMSE': test_rmse, 'NSE': test_nse, 'KGE': test_kge}

        # Check if the stop button is clicked
        if stop_button:
            st.write(f"Training stopped for Transformer model.")

    # Inverse transforming predictions and actual values for plotting
    y_train_inv = scaler.inverse_transform(np.hstack([X_train[:, -1, :], y_train.reshape(-1, 1)]))[:, -1]
    y_test_inv = scaler.inverse_transform(np.hstack([X_test[:, -1, :], y_test.reshape(-1, 1)]))[:, -1]
    lstm_preds_train_inv = scaler.inverse_transform(np.hstack([X_train[:, -1, :], lstm_model.predict(X_train).reshape(-1, 1)]))[:, -1] if train_lstm else None
    transformer_preds_train_inv = scaler.inverse_transform(np.hstack([X_train[:, -1, :], transformer_model.predict(X_train).reshape(-1, 1)]))[:, -1] if train_transformer else None
    lstm_preds_test_inv = scaler.inverse_transform(np.hstack([X_test[:, -1, :], lstm_model.predict(X_test).reshape(-1, 1)]))[:, -1] if train_lstm else None
    transformer_preds_test_inv = scaler.inverse_transform(np.hstack([X_test[:, -1, :], transformer_model.predict(X_test).reshape(-1, 1)]))[:, -1] if train_transformer else None

# Display evaluation metrics
    st.write(f"{selected_basin} - Training Evaluation Metrics:")
    st.write(train_evaluation_metrics)
    st.write(f"{selected_basin} - Testing Evaluation Metrics:")
    st.write(test_evaluation_metrics)

    # Plotting results
    train_dates = pd.date_range(start=basin_data.index[seq_length], periods=len(X_train), freq='D')
    test_dates = pd.date_range(start=basin_data.index[len(X_train) + seq_length - 1], periods=len(X_test), freq='D')
    plot_results(basin_data, train_dates, y_train_inv, lstm_preds_train_inv, transformer_preds_train_inv)
    plot_results(basin_data, test_dates, y_test_inv, lstm_preds_test_inv, transformer_preds_test_inv)

# Animation
animation_button = st.sidebar.button('Play Animation')
if animation_button:
    if isinstance(selected_basins, list) and len(selected_basins) > 0:
        for selected_basin in selected_basins:
            filtered_df = data[data['Basin'] == selected_basin]
            
            if not filtered_df.empty:
                available_years = sorted(filtered_df.index.year.unique())
                
                # Placeholder for the plot
                plot_placeholder = st.empty()
                
                with plot_placeholder.container():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for year in available_years:
                        year_data = filtered_df[filtered_df.index.year == year]
                        if not year_data.empty:
                            ax.clear()
                            if 'Q' in year_data.columns:
                                ax.plot(year_data.index, year_data['Q'], label='Flow Rate (Q)')
                            if 'P' in year_data.columns:
                                ax.plot(year_data.index, year_data['P'], label='Precipitation (P)')
                            ax.set_title(f"Data for {selected_basin} Basin in {year}")
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Value')
                            ax.legend()
                            ax.xaxis.set_major_locator(mdates.MonthLocator())
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                            
                            # Update the plot in the placeholder
                            plot_placeholder.pyplot(fig)
                            time.sleep(0.5)  # Adjust as needed for animation speed
                    
                    st.success(f"Animation completed for {selected_basin} Basin.")
            else:
                st.warning(f"No data available for {selected_basin} Basin.")
    else:
        st.warning("Please select at least one basin for the animation.")

# Main content
if train_button:
    if isinstance(selected_basins, list) and len(selected_basins) > 0:
        # Training and evaluating for each selected basin
        for i, selected_basin in enumerate(selected_basins, start=1):
            st.subheader(f"Basin {i}: {selected_basin}")
            basin_data = data[data['Basin'] == selected_basin].copy()
            
            # Basic statistical analysis
            with st.expander(f"{selected_basin} - Basic Statistical Analysis"):
                st.write(f"Mean Temperature: {basin_data['T'].mean():.2f}")
                st.write(f"Mean Precipitation: {basin_data['P'].mean():.2f}")
                st.write(f"Mean Discharge: {basin_data['Q'].mean():.2f}")
                st.write(f"Temperature Standard Deviation: {basin_data['T'].std():.2f}")
                st.write(f"Precipitation Standard Deviation: {basin_data['P'].std():.2f}")
                st.write(f"Discharge Standard Deviation: {basin_data['Q'].std():.2f}")
                st.write(f"Discharge Quantiles: {basin_data['Q'].quantile([0.25, 0.5, 0.75])}")
            
            # Data visualization
            with st.expander(f"{selected_basin} - Data Visualization"):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(basin_data.index, basin_data['Q'], label='Discharge')
                ax.set_xlabel('Date')
                ax.set_ylabel('Discharge')
                ax.legend()
                st.pyplot(fig)
            
            # Normality test for precipitation values
            with st.expander(f"{selected_basin} - Normality Test for Precipitation"):
                precipitation_data = basin_data['P'].dropna().sample(min(1000, len(basin_data)))
                stat, p = shapiro(precipitation_data)
                if p > 0.05:
                    st.write(f"Precipitation values for {selected_basin} Basin are likely from a normal distribution.")
                else:
                    st.write(f"Precipitation values for {selected_basin} Basin are likely not from a normal distribution.")
            
            # Model training and evaluation
            with st.spinner(f"Training and evaluating models for {selected_basin}..."):
                train_and_evaluate(basin_data, selected_basin)
    else:
        st.warning('Please select at least one basin before training.')
else:
    st.write('Click the "Train Model" button in the sidebar to start training.')