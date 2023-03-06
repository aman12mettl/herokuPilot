from flask import Flask,render_template,Response, jsonify, request
import cv2
import matplotlib.pyplot as plt 
import os 
from PIL import Image
#from import_modules import * 
from metrics import * 
import sys 
import glob 
import pandas as pd 
import numpy as np 
from tensorflow.keras.models import Sequential, Model, load_model

import tensorflow as tf
import tensorflow_addons as tfa

# from metrics import * 
from PIL import Image 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, fbeta_score


state = "PREDICT!!"
image_value = None 
true_label = None
models = []
app=Flask(__name__)
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def load_models(path):
    # path: /models/model_exp_0/ 
    # path models
    global models 
    for i in range(5):
        #model_last = os.path.join(path, f"fold_fold5_{i}_last_model")
        model_last = os.path.join(path, f"fold_fold5_{i}_last_model_weights.h5")
        #model_last="models\model_exp_0\\fold_fold5_0_best_model_weights.h5"
        weights_best = os.path.join(path, f"fold_fold5_{i}_best_model_weights.h5")
        model = load_model(model_last ,custom_objects = {'StatefullBinaryFBeta': StatefullBinaryFBeta(name="fbeta1", beta=0.5),  'Addons>FBetaScore': tfa.metrics.FBetaScore(num_classes=1, beta=0.5)})
        #model.load_weights(weights_best) 
        models.append(model)
    print(len(models), "models loaded")

def prep(img):
    #img = Image.open(pj)
    img= img.resize((224,224))
    img = np.array(img)
    #img = img.reshape((1,224,224,3))
    img = img*1.0/255
    return img

def predict(ig):
    global models, state, true_label
    state = "Predicting..."
    # save ig 
    if true_label == "Real":
        # save it in real folder 
        ig.save(os.path.join(os.path.join("generated_data", "real"), str(time.time())+".png"))
    else:
        # save it in spoof folder 
        ig.save(os.path.join(os.path.join("generated_data", "spoof"), str(time.time())+".png"))
    # cut it into three 
    np_frame = np.array(ig)
    height_full , width_full = np_frame.shape[:-1]
    cut_left = np_frame[:, :height_full,:].copy()
    cut_middle = np_frame[:, int(width_full/2 - (height_full/2)) : int(width_full/2 + (height_full/2)), : ].copy()
    cut_right = np_frame[:, -1 * height_full:, : ].copy()
    cut_left = Image.fromarray(cut_left) 
    cut_middle = Image.fromarray(cut_middle) 
    cut_right = Image.fromarray(cut_right) 

    # cut_left.save("static/1.jpg")
    # cut_middle.save("static/2.jpg")
    # cut_right.save("static/3.jpg")

    image_list = [cut_left, cut_middle, cut_right]
    image_list = [prep(i) for i in image_list]

    test = np.array(image_list)
    preds = []
    thresh = 0.9696969696969697
    # thresh  = 0.5
    # thresh = 0.98989898989899
    # thresh = 0.3
    #thresh = 0.5
    #thresh = 0.979771699568238
    thresh = 0.9
    for model in models:
        preds.append(model.predict(test) )
    preds = np.array(preds) 

    # Alternate way
    preds2 = preds.copy()
    preds2 = (preds2 > thresh).astype(int)
    preds2 = np.sum(preds2, axis=0)
    print("This should have shape of 3", preds2.shape)
    preds2 = (preds2>=2).astype(int)
    preds2 = np.sum(preds2)
    print("What it says", preds2)

    preds = preds**2 
    preds = preds/2 
    print(preds)
    preds= np.sum(preds, axis=0)
    print("Axis:")
    print(preds)
    print()
    preds = (preds>thresh).astype(int)
    print(preds)
    nos = np.sum(preds)
    if nos >= 2:
        state = "Spoof"
    else:
        state = "Real"



picfolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picfolder

def generate_frames():
    global image_value
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            image_value = frame
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    global state 
    state = "PREDICT!!"
    return render_template('index.html', user_image="/static/placeholder1.jpg")

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image_show')
def image_show():
    global image_value
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], "pic1.jpg")
    return Response("index.html", user_image = "/static/current_image.jpg")

import time

@app.route('/_stuff', methods=['GET'])
def stuff():
    global state
    #print("This is time :", time.time())
    # Rotate 
    if state == "PREDICT!!":
        state == ""
    elif state == "":
        state == "PREDICT!!"
    return jsonify(result=state)


@app.route('/refresh', methods=['POST'])
def refresh():
    global state 
    state = "PREDICT!!"

    return render_template('index.html', user_image = "/static/placeholder1.jpg", previous_data="Real")


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    global state ,image_value
    global true_label
    
    true_label=list(request.form.values())[0]
    print("true label", true_label)
    


    if state == "Predicting...":
        return render_template('index.html', user_image = "/static/current_image.jpg", previous_data=true_label)

    else:
        state = "Predicting..."
        ig = image_value[:,:,::-1]
        ig= Image.fromarray(ig, 'RGB')
        print("image size", ig.size)
        #ig2 = ig.resize((320,240))
        ig.save("static/current_image.jpg")
        #render_template('index.html', user_image = "/static/pic2.jpg")
        predict(ig)
        return render_template('index.html', user_image = "/static/current_image.jpg", previous_data=true_label)




if __name__=="__main__":
    load_models("models\model_exp_0")
    app.run(debug=False)