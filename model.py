import pickle
from pydoc import describe
from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.simplefilter("ignore")

df_symptom_severity = pd.read_csv('Data/Symptom-severity.csv')
df_symptom_severity['Symptom'] = df_symptom_severity['Symptom'].str.replace('_', ' ')

df_disease_description = pd.read_csv('Data/symptom_Description.csv')
df_disease_preacaution = pd.read_csv('Data/symptom_precaution.csv')

def predict(s1, s2, s3, s4='vomiting', s5='vomiting', s6='vomiting', s7='vomiting'):
    l = [s1, s2, s3, s4, s5, s6, s7]
    print(l)
    
    x = np.array(df_symptom_severity['Symptom'])
    y = np.array(df_symptom_severity['weight'])
    
    for i in range(len(l)):
        for j in range(len(x)):
            if (l[i] == x[j]):
                l[i] = y[j]
    
    res = [l]
    return res


def predictApi(s1, s2, s3, s4='vomiting', s5='vomiting', s6='vomiting', s7='vomiting'):
    pickled_model = pickle.load(open('RFC_symptoms.pkl', 'rb'))
    predictVar = predict(s1, s2, s3, s4, s5, s6, s7)
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

result = predictApi('stomach pain', 'acidity', 'vomiting')
returned = precautions(result)
descrption = diseaseDescription(result)

print("Disease: " + result + ". Description: " + descrption)
print(returned)
