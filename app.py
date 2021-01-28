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

marine_kwrds_direct = {"marine insurance"}
marine_re = re.compile("|".join(marine_kwrds_direct))


schema =  {
                "fields": [
                    {
                        "name": "index",
                        "type": "integer"
                    },
                    {
                        "name": "ComplaintDesciptionTranslatedEn",
                        "type": "string"
                    }
                ],
                "primaryKey": [
                    "index"
                ]
            }
ins_type_dict = {'Id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}, 'TitleEn': {0: 'Marine Insurance', 1: 'Medical Insurance', 2: 'Property Insurance', 3: 'Vehicle Insurance', 4: 'Other', 5: 'Life Insurance'}, 'TitleAr': {0: 'تأمين بحري', 1: 'تأمين صحي', 2: 'تأمين الممتلكات', 3: 'تأمين المركبات', 4: 'أخرى', 5: 'تأمين الأشخاص أو تكوين الأموال'}}
df_instype_db = pd.DataFrame.from_dict(ins_type_dict)


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
    logging.info('Entry to categorize function')
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
    elif marine_re.search(description):
        return "Marine Insurance"
    else:
        return "Other"
@app.route("/", methods=['GET', 'POST'])
def home():
    logging.info('Entry to home decorator')
    message = '''Hello, this is a sample Python Web App running on Flask Framework for Complaints Categorization !\n
                 Sample JSON input post request \n 
                 {
  "0": {
    "Id": 10401,
    "ComplaintDesciptionTranslatedEn": "Procrastination in the start of repairing the car and not providing an alternative car to the customer according to the new law as it is damaged and caused by its insurance on the new document "
  },
  "1": {
    "Id": 10402,
    "ComplaintDesciptionTranslatedEn": "I provided health insurance to my wife more than 20 days ago at The Guarantee Insurance Company and informed the insurance company after 3 days that the transaction was rejected but they withdrew the amount from my account and filed more than one complaint with them but without result please help me to recover the melg and thank you"
  }
               '''
    return message

@app.route('/categorize', methods=['GET', 'POST'])
def add_message():
    logging.info('Entry to add_message decorator')
    try:
        content = request.get_json(silent=True)
        header = request.headers.get('key')

        if content is None:
            empty_dict = {}
            empty_dict['data'] = []
            empty_dict['status'] = 'failure'
            empty_dict['is_success'] = False
            empty_dict['message'] = "Input data is Empty/Incorrect. Please check the format"
            return json.dumps(empty_dict)
        if header == "TESTKEY":
            pass
        else:
            invali_key_dict = {}
            invali_key_dict['data'] = []
            invali_key_dict['status'] = 'failure'
            invali_key_dict['is_success'] = False
            invali_key_dict['message'] = "Invalid Key"
            return json.dumps(invali_key_dict)

        content['schema'] = schema
        content = json.dumps(content)
        df_input = pd.read_json(content,orient='table')

        df_input['Complaint'] = df_input['ComplaintDesciptionTranslatedEn']
        df_cat = preprocessing(df_input)
        df_cat["cat_pred"] = df_cat.ComplaintDesciptionTranslatedEn.apply(lambda x: categorize(x))

        df_cat = pd.merge(df_cat, df_instype_db, left_on='cat_pred',right_on='TitleEn',how='left')

        dic_out = json.loads(df_cat[['Complaint', 'cat_pred','Id','TitleEn','TitleAr']].to_json(orient='table'))


        keys_return = ['data']

        dic_out = dict((k, dic_out[k]) for k in keys_return)
        dic_out['status'] = 'success'
        dic_out['is_success'] = True
        dic_out['message'] = 'Category prediction successful'
        return json.dumps(dic_out)
    except :
        fail_dict = {}
        fail_dict['data'] = []
        fail_dict['status'] = 'failure'
        fail_dict['is_success'] = False
        fail_dict['message'] = "Error in Processing Input. Check Input json input format"
        return json.dumps(fail_dict)
