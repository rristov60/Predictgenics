import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import altair as alt
import pickle
from unittest import result
import pandas as pd
import numpy as np


def predict(symptoms):

    if(len(symptoms) < 7):
        i = len(symptoms)
        while(i < 7):
            symptoms.append('vomiting')
            i = i + 1
    
    x = np.array(df_symptom_severity['Symptom'])
    y = np.array(df_symptom_severity['weight'])
    
    for i in range(len(symptoms)):
        for j in range(len(x)):
            if (symptoms[i] == x[j]):
                symptoms[i] = y[j]
    
    res = [symptoms]
    return res

def predictAPI(symptoms):
    pickled_model = pickle.load(open('randomForestClassificator.pkl', 'rb'))
    predictVar = predict(symptoms)
    return pickled_model.predict(predictVar)[0]

def precautions(disease):
    x = np.array(df_disease_preacaution['Disease'])
    thePrecautions = []
    precaution1 = np.array(df_disease_preacaution['Precaution_1'])
    precaution2 = np.array(df_disease_preacaution['Precaution_2'])
    precaution3 = np.array(df_disease_preacaution['Precaution_3'])
    precaution4 = np.array(df_disease_preacaution['Precaution_4'])

    for i in range(len(x)):
        if(x[i] == disease):
            print(x[i])
            thePrecautions.append(precaution1[i])
            thePrecautions.append(precaution2[i])
            thePrecautions.append(precaution3[i])
            thePrecautions.append(precaution4[i])
    
    return thePrecautions

def diseaseDescription(disease):
    x = np.array(df_disease_description['Disease'])
    descriptions = df_disease_description['Description']

    for i in range(len(x)):
        if(x[i] == disease):
            return descriptions[i]
    
    return ''

st.set_page_config(
    page_title="Predictgenics", page_icon="💉", initial_sidebar_state="expanded"
)

st.write(
    """
# 💉 Predictgenics
University project app that predicts disease with given symptoms
"""
)

# For production
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # For production

ab_default = None
result_default = None

df_symptom_severity = pd.read_csv('Data/Symptom-severity.csv')
df_symptom_severity['Symptom'] = df_symptom_severity['Symptom'].str.replace('_', ' ')

df_disease_description = pd.read_csv('Data/symptom_Description.csv')
df_disease_preacaution = pd.read_csv('Data/symptom_precaution.csv')


st.markdown("### Experienced Symptoms")
with st.form(key="my_form"):
    symptoms = st.multiselect(
        "Symptoms",
        options=df_symptom_severity['Symptom'].str.title(),
        help="Provide symptoms that you are experiencing. The number of provided symbols should at least **3** but no more than **7** !",
        default=ab_default,
    )

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    if len(symptoms) < 3:
        st.warning("You have to select at least 3 symptoms")
    elif len(symptoms) > 7:
        st.warning("You have to select at most 7 symptoms")
    else:
        for i in range(len(symptoms)):
           symptoms[i] = symptoms[i].lower()
        disease = predictAPI(symptoms)
        description = diseaseDescription(disease)
        precaution = precautions(disease)

        disease = disease.title()

        for i in range(len(precaution)):
           precaution[i] = precaution[i].title()

        st.write(
            """
                It is predicted: """, disease, """. What is """, disease, """ ?

                """, description, """
                
                Precautions:
                """, precaution, """
            """
        )

        print(disease)
        print(description)
        print(precaution)


