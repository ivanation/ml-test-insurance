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
rfr_model = RandomForestRegressor(max_features=3, n_estimators=128)
rfr_model.fit(scaled_X_train, y_train)

def preprocess_input(age, sex, bmi, children, smoker):
  data = pd.DataFrame({"age":[age],"sex":[sex],"bmi":[bmi],"children":[children],"smoker":[smoker]})
  return data

def predict_insurance_charge(data):
  prediction = rfr_model.predict(data)
  return prediction

st.title("Insurance Charge Estimation")
st.sidebar.title("User Input")
age = st.sidebar.slider("Age",20,100,step=1,value=30)
sex = st.sidebar.selectbox("Sex",[0,1], format_func = lambda x: "Male" if x==0 else "Female")
bmi = st.sidebar.slider("BMI",10.0,40.0,step=0.1,value=20.0)
children = st.sidebar.slider("Number of children",0,10,step=1,value=0)
smoker = st.sidebar.selectbox("Smoker",[0,1], format_func = lambda x: "No" if x==0 else "Yes")

input_data = preprocess_input(age, sex, bmi, children, smoker)
prediction = predict_insurance_charge(input_data)

st.subheader("Estimated Insurance Charge:")
result_placeholder = st.empty()
result_placeholder.write(prediction[0])

#####################

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
