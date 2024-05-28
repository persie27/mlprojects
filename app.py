# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# def index():
#     st.title("Home Page")
#     st.write("Welcome to the predictor app!")
#     st.write("Please fill out the form below:")

#     gender = st.selectbox("Gender", ["Male", "Female"])
#     ethnicity = st.selectbox("Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
#     parental_education = st.selectbox("Parental Level of Education", ["Some High School", "High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"])
#     lunch = st.selectbox("Lunch", ["Standard", "Free/Reduced"])
#     test_prep_course = st.selectbox("Test Preparation Course", ["None", "Completed"])
#     reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
#     writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

#     if st.button("Predict"):
#         data = CustomData(
#             gender=gender,
#             race_ethnicity=ethnicity,
#             parental_level_of_education=parental_education,
#             lunch=lunch,
#             test_preparation_course=test_prep_course,
#             reading_score=reading_score,
#             writing_score=writing_score
#         )

#         pred_df = data.get_data_as_data_frame()
#         # print(pred_df.head(1))
#         predict_pipeline = PredictPipeline()
#         results = predict_pipeline.predict(pred_df)

#         st.write("Predicted Result:")
#         st.write(results[0])

# # Run the app
# if __name__ == "__main__":
#     index()


from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        