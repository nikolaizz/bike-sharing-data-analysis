import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import datetime
from babel.numbers import format_currency
sns.set(style='dark')

st.set_page_config(
    page_title="Bike Rental Dashboard",
    page_icon=":bike:",
    initial_sidebar_state="collapsed"
)

def create_hourly_data(df):
  hourly_data = df.groupby(by='Hour').agg({
    'Casual': 'mean',
    'Registered': 'mean',
  }).reset_index()

  return hourly_data

def create_working_day_data(df):
  working_day_data = df.groupby(by='Working Day').agg({
    'Casual': 'mean',
    'Registered': 'mean',
  }).reset_index()

  return working_day_data

def create_seasonal_data(df):
  seasonal_data = df.groupby(by='Season').agg({
    'Casual': 'mean',
    'Registered': 'mean',
  }).reset_index()

  return seasonal_data

def create_trend_data(df):
  trend_data = df.groupby(by=['Year', 'Month']).agg({
    'Total': 'sum'
  }).reset_index()
  trend_data['Date'] = pd.to_datetime(trend_data['Year'].astype(str) + '-' + trend_data['Month'].astype(str), format='%Y-%B')
  trend_data = trend_data.sort_values(by='Date')

  return trend_data

def create_anomaly_data(df):
  daily_data = df.groupby('Date').agg({
    'Total': 'sum'
}).reset_index()

  daily_data['change'] = daily_data['Total'].diff()
  daily_data['change'].fillna(0, inplace=True)

  threshold = daily_data['change'].std() * 2
  anomalies = daily_data[abs(daily_data['change']) > threshold]
  
  return daily_data, anomalies, threshold

day_df_clean = pd.read_csv('dashboard/day_clean.csv')
hour_df_clean = pd.read_csv('dashboard/hour_clean.csv')

with st.sidebar:
  st.image('https://allvectorlogo.com/img/2017/07/capital-bikeshare-logo.png')
  st.title('Capital Bikeshare')

  col1, col2 = st.columns(2)
  with col1:
    st.markdown("[Dataset Source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)")
  with col2:
    st.markdown("[Source Code](https://github.com/nikolaizz/bike-sharing-data-analysis)")

hourly_data = create_hourly_data(hour_df_clean)
working_day_data = create_working_day_data(day_df_clean)
seasonal_data = create_seasonal_data(day_df_clean)
trend_data = create_trend_data(day_df_clean)
daily_data, anomalies, threshold = create_anomaly_data(hour_df_clean)

st.header("Bike Rental Dashboard :bike:")

st.subheader("Hourly Data")

fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='Hour', y='Registered', data=hourly_data, ax=ax, label='Registered', color='blue')
sns.barplot(x='Hour', y='Casual', data=hourly_data, ax=ax, label='Casual', color='orange')

ax.set_xlabel('Hour')
ax.set_ylabel('Average Count')
ax.set_title('Average Bike Rental Count by Hour')
ax.legend()
st.pyplot(fig)

st.divider()

st.subheader("Working Day Data")
fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='Working Day', y='Registered', data=working_day_data, ax=ax, label='Registered', color='blue')
sns.barplot(x='Working Day', y='Casual', data=working_day_data, ax=ax, label='Casual', color='orange')

ax.set_xlabel('Working Day')
ax.set_ylabel('Average Count')
ax.set_title('Average Bike Rental Count by Working Day')
ax.legend()
st.pyplot(fig)

st.divider()

st.subheader("Seasonal Data")
fig, ax = plt.subplots(figsize=(16, 8))

sns.barplot(x='Season', y='Registered', data=seasonal_data, ax=ax, label='Registered', color='blue')
sns.barplot(x='Season', y='Casual', data=seasonal_data, ax=ax, label='Casual', color='orange')

ax.set_xlabel('Season')
ax.set_ylabel('Average Count')
ax.set_title('Average Bike Rental Count by Season')
ax.legend()
st.pyplot(fig)

st.divider()

st.subheader("Trend Data")
fig, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(data=trend_data, x='Date', y='Total', marker='o', color='blue')

plt.title('Bike Rental Trend by Month (January 2011 - December 2012)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Total Count')
plt.grid(True)
st.pyplot(fig)

st.divider()

st.subheader("Anomaly Detection")

col1, = st.columns(1)

with col1:
  anomaly_sum = len(anomalies)
  st.metric(label='Detected Anomaly', value=anomaly_sum)

fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(data=daily_data, x='Date', y='change', ax=ax, label='Daily Change', color='blue')

ax.scatter(anomalies['Date'], anomalies['change'], color='red', label='Anomaly', s=50)

ax.axhline(threshold, color='green', linestyle='--', label='Positive Threshold')
ax.axhline(-threshold, color='orange', linestyle='--', label='Negative Threshold')

ax.set_title('Daily Change in Bike Rental Count with Anomalies')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Change in Count')

ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption("Dataset by Capital Bikeshare System, Washington D.C., USA")
