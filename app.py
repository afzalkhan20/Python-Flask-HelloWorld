import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
import json
import flask
from flask import request, jsonify
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import logging

stop_words = stopwords.words('english')

import en_core_web_sm
nlp = en_core_web_sm.load()


app = flask.Flask(__name__)
app.config["DEBUG"] = True
#----------------------Configuration Starts---------------------#

vehicle_keywords = {"car","accident","vehicle","comprehensive","chasis","rta","traffic","police","rent","nissan",
                    "honda","toyota","parts","repair","damage","Chevrolet"}
vehicle_kwrds_len = len(vehicle_keywords)
vehicle_kwrds_direct = {"car insurance","vehicle insurance"}
vehicle_re = re.compile("|".join(vehicle_kwrds_direct))

property_keywords = {"wall","property","steal","house","fire","warehouse"}
property_kwrds_len = len(vehicle_keywords)
property_kwrds_direct = {"property insurance"}
property_re = re.compile("|".join(property_kwrds_direct))


medical_keywords = {"hospital","treatment","clinic","lab","disease","injury","death","operation"}
medical_kwrds_len = len(medical_keywords)
medical_kwrds_direct = {"medical insurance","health insurance"}
medical_re = re.compile("|".join(medical_kwrds_direct))

life_keywords = {"life","death"}
life_kwrds_len = len(life_keywords)
life_kwrds_direct = {"life insurance"}
life_re = re.compile("|".join(life_kwrds_direct))
#-----------------------------Configuration Ends---------------------#

# function to remove stopwords
def remove_stopwords(des):
    logging.info('Entry to remove_stopwords')
    des_new = " ".join([i for i in des if i not in stop_words])
    return des_new

def preprocessing(df_preprocess):
    logging.info('Entry to preprocessing')
    df_preprocess['ComplaintDesciptionTranslatedEn'] = df_preprocess['ComplaintDesciptionTranslatedEn'].str.lower()
    # remove unwanted characters, numbers and symbols
    df_preprocess['ComplaintDesciptionTranslatedEn'] = df_preprocess['ComplaintDesciptionTranslatedEn'].str.replace("[^a-zA-Z#]", " ")

    # remove short words (length < 3)
    df_preprocess['ComplaintDesciptionTranslatedEn'] = df_preprocess['ComplaintDesciptionTranslatedEn'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    # remove stopwords from the text
    df_preprocess['ComplaintDesciptionTranslatedEn'] = [remove_stopwords(r.split()) for r in df_preprocess['ComplaintDesciptionTranslatedEn']]

    df_preprocess['ComplaintDesciptionTranslatedEn'] = df_preprocess['ComplaintDesciptionTranslatedEn'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))

    return df_preprocess


def categorize(description):
    logging.info('Entry to categorize')
    # print(description)
    des_kwrds = set(description.split(" "))

    vehicle_cat_prob = len(des_kwrds.intersection(vehicle_keywords)) / vehicle_kwrds_len
    property_cat_prob = len(des_kwrds.intersection(property_keywords)) / property_kwrds_len
    medical_cat_prob = len(des_kwrds.intersection(medical_keywords)) / medical_kwrds_len
    life_cat_prob = len(des_kwrds.intersection(life_keywords)) / life_kwrds_len

    if ((vehicle_cat_prob > property_cat_prob) and (vehicle_cat_prob > medical_cat_prob)) or (
    vehicle_re.search(description)):
        return "Vehicle Insurance"
    elif (medical_cat_prob > property_cat_prob) or medical_re.search(description):
        return "Medical Insurance"
    elif life_re.search(description):
        return "Life Insurance"
    elif (property_cat_prob > 0) or property_re.search(description):
        return "Property Insurance"
    else:
        return "Other"
@app.route("/", methods=['GET', 'POST'])
def home():
    logging.info('Entry to home decorator')
    return "Hello, this is a sample Python Web App running on Flask Framework!"

@app.route('/categorize', methods=['GET', 'POST'])
def add_message():
    logging.info('Entry to add_message decorator')
    try:
        content = request.get_json(silent=True)
        df_input = pd.DataFrame.from_dict(content, orient="index")
        df_input['Complaint'] = df_input['ComplaintDesciptionTranslatedEn']
        df_cat = preprocessing(df_input)
        df_cat["cat_pred"] = df_cat.ComplaintDesciptionTranslatedEn.apply(lambda x: categorize(x))

    except Exception as exc:
        return "Error in Processing"
    finally:
        return df_cat[['Id','Complaint','cat_pred']].to_json(orient='index')
