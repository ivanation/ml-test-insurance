import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestRegressor

labelencoder = preprocessing.LabelEncoder()
scaler = preprocessing.StandardScaler()
data = pd.read_csv("insurance.csv")
data.drop_duplicates(inplace=True)
data = data[["age","sex","bmi","children","smoker","charges"]]
data['sex'] = labelencoder.fit_transform(data['sex'])
data['smoker'] = labelencoder.fit_transform(data['smoker'])
X = data.drop(columns='charges')
y = data['charges']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
scaled_X_train = scaler.fit_transform(X_train)

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
