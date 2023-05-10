import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

st.write("# Big Data Project  \n _Taxi Price Prediction_\n", "*Year*: **2023**")

st.write("## Query 1  \n Correlation between distance, price, frequency of place as a start point and price for "
         "mile")

query = pd.read_csv("output/q1.csv")[["avg_payed_in", "avg_dist_in", "rides_number_in", "price_for_dist_in"]]
fig = plt.figure(figsize=(8, 6))
sns.heatmap(query.corr(), annot=True)
st.pyplot(fig)

st.write("## Query 2  \n Correlation between distance, price, frequency of place as an end point and price for "
         "mile")

query = pd.read_csv("output/q2.csv")[["avg_payed_out", "avg_dist_out", "rides_number_out", "price_for_dist_out"]]
fig = plt.figure(figsize=(8, 6))
sns.heatmap(query.corr(), annot=True)
st.pyplot(fig)

st.write("## Query 3  \n Frequency of rides between geo points")

query = pd.read_csv("output/q3.csv")
plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(50, 50))
query['count'] = query['count'].apply(np.log2)
query['fare'] = query['fare'].apply(np.log2)
square = query.pivot_table(values='count', index=query.dropoff_location_id, columns=query.pickup_location_id,
                           aggfunc='first')
sns.heatmap(square, cmap="Greens", vmin=1)
st.pyplot(fig)

st.write("Avg price for mile between geo points")

plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(50, 50))
square = query.pivot_table(values='fare', index=query.dropoff_location_id, columns=query.pickup_location_id,
                           aggfunc='first')
sns.heatmap(square, cmap="Greens", vmin=1)
st.pyplot(fig)

st.write("## Query 4  \n Total for ride")

query = pd.read_csv("output/q4.csv")
query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist']] = query[
    ['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist']].apply(np.log)
fig = plt.figure(figsize=(7, 4))
sns.barplot(x='passenger_count', y='value', hue='variable',
            data=pd.melt(query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist', 'passenger_count']],
                         ['passenger_count']))
st.pyplot(fig)

fig = plt.figure(figsize=(7, 4))
sns.lineplot(x='passenger_count', y='value', hue='variable',
             data=pd.melt(query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist', 'passenger_count']],
                          ['passenger_count']))
st.pyplot(fig)