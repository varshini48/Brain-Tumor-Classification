# Load all libraries and packages
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html  # Change this line
from dash import no_update
import base64
import io
from io import BytesIO
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import load_model
def get_color(index):
    colors = ['red', 'blue', 'green', 'purple']  # Customize the colors as needed
    return colors[index % len(colors)]
# Define classes using onehotencoder

classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3]]) 
def names(number):
    if(number == 0):
        return 'a Glioma tumor'
    elif(number == 1):
        return 'a Meningioma tumor'
    elif(number == 2):
        return 'no tumor'
    elif(number == 3):
        return 'a Pituitary tumor'  
#External style sheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    
    html.H1(children='BRAIN TUMOUR DETECTION', style={'textAlign': 'center','background': 'white'
        }),
    
    
    dcc.Markdown('''
                ###### Step 1: Upload the image here
                ###### Step 2: Please wait for the results
    '''),
    
    
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
            
        }, 
        # Allow multiple files to be uploaded
        multiple=True
    ),
    
    html.Div(id='output-image-upload', style={'position':'absolute', 'left':'200px', 'top':'250px'}),
    html.Div(id='prediction', style={'position':'absolute', 'left':'800px', 'top':'310px', 'font-size':'x-large','color':'red','backgroundColor':'yellow'}),
    html.Div(id='prediction2', style={'position':'absolute', 'left':'800px', 'top':'365px', 'font-size': 'x-large','color':'red','backgroundColor':'yellow'}),
    html.Div(id='facts', style={'position':'absolute', 'left':'800px', 'top':'565px', 'font-size': 'large',\
                               'height': '150px', 'width': '400px','background': 'white','color':'red'}),
    html.Div(id='alert-message', style={'position': 'fixed', 'middle': '10px', 'right': '10px', 'font-size': 'large','backgroundColor': 'yellow'}),
    html.Div(id='neurologists-link', style={'position':'absolute', 'left':'800px', 'top':'420px','backgroundColor': 'black'}),
    

 
     

],style={'backgroundImage': 'url("http://getwallpapers.com/wallpaper/full/6/b/3/70048.jpg")', 'backgroundSize': 'cover', 'height': '100vh'})


def parse_contents(contents):

    return html.Img(src=contents, style={'height':'450px', 'width':'450px'})
#call back function

@app.callback([Output('output-image-upload', 'children'), Output('prediction', 'children'), Output('prediction2', 'children'), 
              Output('facts', 'children'),Output('alert-message','children'),Output('neurologists-link', 'children')],
              [Input('upload-image', 'contents')])

def update_output(list_of_contents):  
          
    
    if list_of_contents is not None:
        children = parse_contents(list_of_contents[0]) 
         
        img_data = list_of_contents[0]
        img_data = re.sub('data:image/jpeg;base64,', '', img_data)
        img_data = base64.b64decode(img_data)  
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        
        #Load model, change image to array and predict
        model = load_model('C:/Users/Varshini/Downloads/brain-tumor-classifier-app-master (2)/brain-tumor-classifier-app-master/model_final.h5') 
        dim = (150, 150)
        
        img = np.array(img_pil.resize(dim))
        
        x = img.reshape(1,150,150,3)

        answ = model.predict(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        pred=str(round(answ[0][classification]*100 ,3)) + '% confidence there is ' + names(classification)   
        facts = None
        pred2 = None
        
        # facts about tumour if there is
        if classification==0:
            facts = 'Glioma is a type of tumor that occurs in the brain and spinal cord.\
                    A glioma can affect your brain function and be life-threatening depending on\
                    its location and rate of growth.'
                    
            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
            facts_div = html.Div(facts, style={'color': 'black', 'fontSize': 'large'})
            alert_message = html.Div(['CONSULT DOCTOR IMMEDIATELY!!'], style={'color': 'red', 'fontSize': 'large'})
            neurologists_link = html.A('Find the best neurologists here', href='https://www.clinicspots.com/neurologist/india', target='_blank', style={'color': 'green', 'fontSize': 'large', 'margin-right': '40px'})
            
            
        elif classification==1:
            facts = 'A meningioma is a tumor that arises from the meninges, the membranes that surround your brain.\
                    Most meningiomas grow very slowly, often over many years without causing symptoms.'
            alert_message = html.Div(['CONSULT DOCTOR IMMEDIATELY!!'], style={'color': 'red', 'fontSize': 'large'})
            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
            facts_div = html.Div(facts, style={'color': 'black', 'fontSize': 'large'})
            neurologists_link = html.A('Find the best neurologists here', href='https://www.clinicspots.com/neurologist/india', target='_blank', style={'color': 'green', 'fontSize': 'large', 'margin-right': '40px'})
            
        elif classification==2:
            facts = 'NO NEED TO WORRY \
                   YOU ARE PERFECTLY ALRIGHT !!   '
            facts_div = html.Div(facts, style={'color': 'blue', 'fontSize': 'large','height':'100px','width':'300'})
            alert_message= no_update
            neurologists_link=no_update
            
        elif classification==3:
            facts = 'Pituitary tumors are abnormal growths that develop in your pituitary gland.\
                    Most pituitary tumors are noncancerous (benign) growths that remain in your pituitary\
                    gland or surrounding tissues.'
                    
            
            
            no_tumor = str(round(answ[0][2]*100 ,3))
            pred2 = no_tumor + '% confidence there is no tumor'
            facts_div = html.Div(facts, style={'color': 'black', 'fontSize': 'large'})
            alert_message = html.Div(['CONSULT DOCTOR IMMEDIATELY!!'], style={'color': 'red', 'fontSize': 'large'})
            neurologists_link = html.A('Find the best neurologists here', href='https://www.clinicspots.com/neurologist/india', target='_blank', style={'color': 'green', 'fontSize': 'large', 'margin-right': '40px'})
        else:
            facts=None
            pred2 = None
            alert_message = no_update
            neurologists_link=no_update
            
        return children, pred, pred2, facts_div,alert_message,neurologists_link
    else:
        return (no_update, no_update, no_update, no_update,no_update, no_update)  

if __name__ == '__main__':
    app.run_server(debug=True)
