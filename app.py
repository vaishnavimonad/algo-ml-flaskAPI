#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configparser
import warnings
from argparse import ArgumentParser

warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import logging
from flask import Flask, render_template, request, abort, redirect, url_for, session, jsonify, make_response
from flask_cors import CORS
from wtforms import Form, TextField
from datetime import datetime
import json
import pandas as pd

app = Flask(__name__)
app.app_context().push()
app.secret_key = 'secret'
logger = app.logger
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
app.debug = True
logger.name = 'data_predict'
CORS(app)
config = configparser.ConfigParser()
config.read('config.ini')
logging.basicConfig(filename=config['DATA']['LOG_PATH'], filemode='a', format='%(name)s - %(levelname)s - %(message)s')

from predictive_maintenance.PredictValue import predict_value
predict_value = predict_value()
import os
import redis
from rq import Queue

r = redis.Redis()
q_high = Queue('high', connection=r)
q_mid = Queue('mid', connection=r)
q_low = Queue('low', connection=r)
q_default = Queue('default', connection=r)

model1 = config['DATA']['MODEL1']
model2 = config['DATA']['MODEL2']
input_folder = config['DATA']['input_fl']
output_foler = config['DATA']['output_fl']
timeout= config['DATA']['JOB_TIMEOUT']

def to_float(val):
    try:
        return float(val)
    except ValueError:
        return None
    except TypeError:
        return None


@app.route("/data_predict", methods=['GET', 'POST'])
def get_predict_data():
    start = datetime.now()
    
    header = None
    url = request.url
    status = 0
            
    try:
        modelname = request.args.get("modelname")
        header = modelname
        filename = request.args.get("filename")
        priority = int(request.args.get("priority"))
    except Exception as ex:
        response = jsonify({'error_code': 400, 
                            'error_msg':'missing parameters'})
        return make_response((response))

    out_path = os.path.join(output_foler,"{0}_predictions_{1}".format(modelname, filename))
    try:        
        
        inputfile_path = os.path.join(input_folder,filename)     
        df = pd.read_csv(inputfile_path, na_values=['?'])
        
        df.index = df['curveNb']
        X_test = df.drop(labels=['curveNb','2P_yearMonth'], axis=1)   
       
        if priority == 1:            
            job = q_high.enqueue_call(func=predict_value.predict, args=(X_test, modelname, out_path), timeout=timeout)
            logger.info(f"Task ({job.id}) added to high queue at {job.enqueued_at}")
        
        elif priority == 2:            
            job = q_mid.enqueue_call(func=predict_value.predict, args=(X_test, modelname, out_path), timeout=timeout)
            logger.info(f"Task ({job.id}) added to mid queue at {job.enqueued_at}") 
   
        elif priority == 3:            
            job = q_low.enqueue_call(func=predict_value.predict, args=(X_test, modelname, out_path), timeout=timeout)
            logger.info(f"Task ({job.id}) added to low queue at {job.enqueued_at}")    

        else:          
            job = q_default.enqueue_call(func=predict_value.predict, args=(X_test, modelname, out_path), timeout=timeout)
            logger.info(f"Task ({job.id}) added to default queue at {job.enqueued_at}")    
        
        status = 200
    except KeyError as ex:
        msg = {'error_code':404,
               'error_message':'{}'.format(ex)}
    except Exception as ex:
        msg = {'error_code':520,
               'error_message':'{}'.format(ex)}

    if status == 200:
        msg = {'status_code': 200,
                'status': 'SUCCESS' }
    else:
        msg = {'error_code': 400,
            'error_message': msg }

    time_taken = (datetime.now() - start)
    logger.info('Request served')
    logger.info('time_taken: {}'.format(time_taken))

    response = {'status':msg,'url':url,'output file':os.path.basename(out_path)}
    logger.info(response)
    return jsonify({header:response})


@app.route('/health_check', methods=['GET'])
def health_check():
    return make_response(('SUCCESS', 200))


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    default_port = int(config['DATA']['port'])
    parser = ArgumentParser(description='Pass Data')
    parser.add_argument('-p', '--port', default=default_port, help='port to listen on')
    args = parser.parse_args()
    app.run("0.0.0.0", port=default_port)
