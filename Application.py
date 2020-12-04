import pandas as pd
import pickle
import datetime 
import numpy as np
from tkinter import *
from tkinter import filedialog 
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords


# Function for opening the  
# file explorer window 

def make_prediction():  
    test_df = pd.read_csv(filename) 
    stopword = stopwords.words('english')
    test_df['CleanReview'] = test_df['review_by_patient'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopword]))
    analyzer = SentimentIntensityAnalyzer()
    test_df['Review_Score']= test_df['CleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    test_df['ReviewType'] = test_df['Review_Score'].map(lambda x:int(2) if x>=0.05 else int(1))
    test_df['EfficiencyType'] = test_df['effectiveness_rating'].apply(lambda x: int(2) if x>=7 else int(1) if x<4 else int(0))
    test_input_feature = test_df[['effectiveness_rating','number_of_times_prescribed','ReviewType','EfficiencyType']]
    model = pickle.load(open("pima.pickle.dat", "rb"))
    prediction = model.predict(test_input_feature)
    output = pd.DataFrame()
    output['patient_id'] = test_df['patient_id']
    output['base_score'] = prediction
    output.to_csv('predicted_test_file.csv',index=False)
    
def browseFiles(): 
    global filename
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("Text files", 
                                                        "*.txt*"), 
                                                       ("all files", 
                                                        "*.*"))) 
       
    # Change label contents 
    label_file_explorer.configure(text="File Opened: "+filename) 
    
       
       
                                                                                                   
# Create the root window 
window = Tk() 
   
# Set window title 
window.title('File Explorer') 
   
# Set window size 
window.geometry("500x500") 
   
#Set window background color 
window.config(background = "white") 
   
# Create a File Explorer label 
label_file_explorer = Label(window,  
                            text = "File Explorer using Tkinter", 
                            width = 100, height = 4,  
                            fg = "blue") 
   
       
button_explore = Button(window,  
                        text = "Browse Files", 
                        command = browseFiles)  
   
button_exit = Button(window,  
                     text = "Predict", 
                     command = make_prediction)  
   
# Grid method is chosen for placing 
# the widgets at respective positions  
# in a table like structure by 
# specifying rows and columns 
label_file_explorer.grid(column = 1, row = 1) 
   
button_explore.grid(column = 1, row = 2) 
   
button_exit.grid(column = 1,row = 3) 
   
# Let the window wait for any events 
window.mainloop() 