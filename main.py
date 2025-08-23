import pickle
import streamlit as st
import pandas as pd
from os import path

st.title('Flower species predictor')


petal_length = st.number_input("Choose a petal length", placeholder ="Enter a value between 1.0 and 6.9",min_value =1.0, max_value=6.9,value= None)
petal_width = st.number_input("Choose a petal width", placeholder="Enter a value between 0.1 and 2.5", min_value =0.1, max_value=2.5,value= None)
sepal_length = st.number_input("Choose a sepal length", placeholder="Enter a value between 4.3 and 7.9", min_value =4.3, max_value=7.9,value= None)
sepal_width = st.number_input("Choose a sepal width", placeholder="Enter a value between 2.0 and 4.4", min_value =2.0, max_value=4.4,value= None)

#prepare dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns = ['sepal_length','sepal_width','petal_length','petal_width'])

#using .pkl file, creating ml model named 'iris_predictor'
#loading pickled file from Model
model_path = path.join("Model","iris.pkl")
with open(model_path, "rb") as file:
    iris_predictor = pickle.load(file)

dict_species = {0:'setosa',1:'versicolor',2:'viriginica'}
if st.button("Predict species"):
    if((petal_length==None) or (petal_width==None)
            or (sepal_length==None) or (sepal_width==None)):
        st.write("Please enter all values") # will be executed when any of the values is not entered properly
    else:
        #prediction can be done here. we are expecting a dataframe
        predicted_species = iris_predictor.predict(df_user_input)
        #[predicted_species[0] will give us the value in dataframe.
        #we use that value to find corresponding species from dictionary species
        st.write("the species is", dict_species[predicted_species[0]])
