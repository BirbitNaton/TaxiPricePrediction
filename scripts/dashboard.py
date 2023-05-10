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
st.write("for the dropoff points we see decent correlation of frequency of rides to the point and fare prices - suggests that the more popular a point to get, the more customers have to pay; also we see even stronger one (but negative) between distance to a point and fare price - suggests that the longer the ride, the fewer customers pay per mile. Obviously we can observe a huge correlation between distance and total price, but that.")

st.write("## Query 2  \n Correlation between distance, price, frequency of place as an end point and price for "
         "mile")

query = pd.read_csv("output/q2.csv")[["avg_payed_out", "avg_dist_out", "rides_number_out", "price_for_dist_out"]]
fig = plt.figure(figsize=(8, 6))
sns.heatmap(query.corr(), annot=True)
st.pyplot(fig)
st.write("For the pickup points we still see strong negative correlation between distance from the point and fare price - ride further, pay less mile-wise; and strong positive correlation between price and price per mile. Also, we can notice interestingly low correlation between rides_number_out and both avg_payed_out and price_for_dist_out - that gives us a hit of uselessness of the frequency of rides from a point in general, which on the other hand is too general to state for all the pickup points, so we won't totally discard the feature.")

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
st.write("Here we see the heatmap of ride frequencies from pickup (horizontal) and dropoff locations (vertical). This way we see that some locations are more popular to get to and others from. The bright clusters on the main diagonal denote that a cloud of points is both popular on the way in and out. The bright line on the diagonal shows that many people use taxi within a location. As sometimes prices skyrocket with growth of the ask, we should note that.\n",
    "\n",
    "Another heatmap shows us dependency of fare prices for different routs. To surprise, we can see the dark spots where popular (heatmap above) and simultaneously expensive (bright cells along axis for clusters of values of another) routes cross - that gives us a hint of the fare<->location dependency being non-linear, my guess is any linear regressor would perform poorly on these features, so I reckon forest- or SVM-based models are the choice.\n",
    "\n",
    "Both count and fare features are normalized with log2, as means of noise reduction.")



st.write("## Query 4  \n Total for ride")

query = pd.read_csv("output/q4.csv")
query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist']] = query[
    ['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist']].apply(np.log)
fig = plt.figure(figsize=(7, 4))
sns.barplot(x='passenger_count', y='value', hue='variable',
            data=pd.melt(query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist', 'passenger_count']],
                         ['passenger_count']))
st.pyplot(fig)
st.write("The following barplot shows how mile-wise price changes with number of passenger varying. In average, it shows simultaneous growth, although max and min prices are further from average the fewer passengers there are. This simply shows higher dispersion and thus it changes the confidence interval. All prices are normalized with log for sake of noise reduction.")

fig = plt.figure(figsize=(7, 4))
sns.lineplot(x='passenger_count', y='value', hue='variable',
             data=pd.melt(query[['min_total_for_dist', 'avg_total_for_dist', 'max_total_for_dist', 'passenger_count']],
                          ['passenger_count']))
st.pyplot(fig)

st.write("## Query6 \n Correlation of rides in with months and hours")

query = pd.read_csv("output/q6.csv")
plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(8,6))
square = query.pivot_table(values='count', index=query.hour, columns=query.month, aggfunc='first')
sns.heatmap(square, cmap="Greens")
st.pyplot(fig)

fig = plt.figure(figsize=(8,6))
square = query.pivot_table(values='max_fare', index=query.hour, columns=query.month, aggfunc='first')
sns.heatmap(square, cmap="Greens", vmax=100)
st.pyplot(fig)

fig = plt.figure(figsize=(8,6))
square = query.pivot_table(values='avg_fare', index=query.hour, columns=query.month, aggfunc='first')
sns.heatmap(square, cmap="Greens")
st.pyplot(fig)

st.write("All 3 plots show heatmaps of features aggregated against month and hour. On the 1st plot we see a spike on March and a dark line at night, suggesting fewer people at that hour regardless of the month (except said spike on March), yields mediated correlation with price. Second heatmap shows max_fare. The feature is noisy, but we can see lower values at night, supports previous point. On the third plot we see average fare drop late at night and spikes around 4pm and 5am. Taking closer look, we can notice how cells on spring and later autumn are ever so slightly brighter, so seasons might matter")

st.write("## GBT model prediction results")
gbtr = pd.read_csv("output/model2_predictions.csv").reset_index()
st.write(gbtr.head())
st.write("Metrics. RMSE: 4.75, R^2: 0.87")

plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(15, 10))
sns.lineplot(x='index', y='value', hue='variable', data=pd.melt(gbtr, ['index']), linewidth=0.1, alpha=0.7)
st.pyplot(fig)

st.write("The first graph shows GBT Regressor's predictions (blue) along with real test values (orange). It shows decent overall results, despite some abysmal gaps on big spikes as the dataset still contains some noise, especially it concerns abnormally big values")

plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(15, 10))
temp = gbtr.copy()
temp['diff'] = temp['prediction']-temp['total_amount']
sns.lineplot(x='index', y='diff', data=temp, linewidth=0.1)
st.pyplot(fig)

st.write("Second graph shows the exact difference of prediction and real text values. Here we clearly see places where the model hiccups. Again, it's mostly the problem of the spikes. For an empirical reference you can consider prices in the range $40-80 to be 'normal' - that shows the scale.")

st.write("## Random Forest model prediction results")
rfr = pd.read_csv("output/model1_predictions.csv").reset_index()
st.write(rfr.head())
st.write("Metrics. RMSE: 4.42, R^2: 0.88")

plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(15, 10))
sns.lineplot(x='index', y='value', hue='variable', data=pd.melt(rfr, ['index']), linewidth=0.1, alpha=0.7)
st.pyplot(fig)

plt.gcf().set_dpi(600)
fig = plt.figure(figsize=(15, 10))
temp = rfr.copy()
temp['diff'] = temp['prediction']-temp['total_amount']
sns.lineplot(x='index', y='diff', data=temp, linewidth=0.1, alpha=0.7)
st.pyplot(fig)

st.write("These two graphs feature the same logic as the first and second, but for random forest regressor. Here we see that it's performance on spikes is better and that it arguably closer oscillates near the 0.")