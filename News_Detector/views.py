from django.http import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
import pickle
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import re
import io
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
#import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd

 

model_path=r'News_Detector\saved_models\news_ml_classifier.pkl'
vector_path=r'News_Detector\saved_models\vectorized_weight.pkl'


def index(request):
    #return render(request,'index.html',{})
    if request.method=='POST':
        input_text=request.POST.get('text_input',' ')

    
        #print(input_text)
    
        with open(model_path,'rb') as file:
            model=pickle.load(file)
        with open(vector_path,'rb') as f:
            vectorize=pickle.load(f)
        #function for data cleaning and lemmatization to covert word to base word with actual meaning
        def lemmatize_func(text):
            text=text.lower()
            text=re.sub(r'[^a-zA-Z\s]','',text)
            tokenize_text=word_tokenize(text)
            filtered_token=[token for token in tokenize_text if token not in stopwords.words('english')]
            lemmatize_obj=WordNetLemmatizer()
            lemmatize_text=[lemmatize_obj.lemmatize(word) for word in filtered_token]
            cleaned_text=' '.join(lemmatize_text)
            return cleaned_text

        clean_data=lemmatize_func(input_text)

        listed_text=[clean_data]
            

        vectorized_text =vectorize.transform(listed_text)

        
        prediction_prob=model.predict_proba(vectorized_text)
        prediction=model.predict(vectorized_text)
        
        reliability=round(prediction_prob[0][1],2)
        print(f"The prediction is: {prediction}")
        print(f"Prediction type:{type(prediction)}")
        pr=None
        if(prediction[0])==1:
            print("The News is Reliable")
            pr='Reliable'
        else:
            print("Not reliable")
            pr='Not Reliable'


        all_data={'prediction': prediction[0],
                  'reliability':reliability,
                  'pr':pr}
        
    

    
        return render(request,'index.html',all_data)
    else:
        return render(request,'index.html',{})


    
def about(request):
    return render(request,'about.html',{})


def contact(request):
    return render(request,'contact.html',{})

def calculate_roc(y_train,y_prob):
    fpr,tpr,_=metrics.roc_curve(y_train,y_prob)
    return fpr.tolist(), tpr.tolist()

def news_detect(request):
    print("Hi in newsd")
    if request.method=="POST":
        csv_file=request.FILES['file']
        
        news_df=pd.read_csv(csv_file)
        print(news_df.head())
        
        news_df=news_df.fillna('')
        
        news_df['content']=news_df['text']+news_df['title']
        
        print(news_df['content'])
        
        
        def lemmatize_func(text):
            text=text.lower()
            text=re.sub(r'[^a-zA-Z\s]','',text)
            tokenize_text=word_tokenize(text)
            filtered_token=[token for token in tokenize_text if token not in stopwords.words('english')]
            lemmatize_obj=WordNetLemmatizer()
            lemmatize_text=[lemmatize_obj.lemmatize(word) for word in filtered_token]
            cleaned_text=' '.join(lemmatize_text)
            return cleaned_text
        
        news_df['content']=news_df['content'].apply(lemmatize_func)
        
        x_data=news_df['content']
        y_target=news_df['label']
        
        vectorization=TfidfVectorizer()
        vectorization.fit(x_data)
        x_data=vectorization.transform(x_data)
        
        x_train,x_test,y_train,y_test=train_test_split(x_data,y_target,test_size=0.2,stratify=y_target,random_state=2)
        
        classifier_model=LogisticRegression()
        classifier_model.fit(x_train,y_train)
        
        y_pred = classifier_model.predict(x_test)

        # Calculate accuracy

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        model_ytrain_pred=classifier_model.predict(x_train)
        model_train_cmt=metrics.confusion_matrix(y_train,model_ytrain_pred)
        
        model_test_pred=classifier_model.predict(x_test)
        model_test_cmt=metrics.confusion_matrix(y_test,model_test_pred)
        print(model_train_cmt)
        print(model_test_cmt)
        
        
        #roc
        y_train_prob=classifier_model.predict_proba(x_train)[:,1]
        fpr_tr,tpr_tr=calculate_roc(y_train,y_train_prob)
        
        
        y_test_prob=classifier_model.predict_proba(x_test)[:,1]
        fpr_tst,tpr_tst=calculate_roc(y_test,y_test_prob)
        all_data={'acc':accuracy,
                  'msg':'Trained Successfully',
                  'confusion_matrix_train':model_train_cmt.tolist(),
                  'confusion_matrix_test':model_test_cmt.tolist(),
                  'fptr':fpr_tr,
                  'tptr':tpr_tr,
                  'fptst':fpr_tst,
                  'tptst':tpr_tst}
        
        
        with open('News_Detector\\saved_models\\user_model.pkl', 'wb') as model_file:
            pickle.dump(classifier_model, model_file)
        
        with open('News_Detector\\saved_models\\user_vectorized_weight.pkl','wb') as f:
            pickle.dump(vectorization,f)
        return render(request,'news_detect.html',all_data)
    


        
        
    else:
        return render(request,'news_detect.html',{})


def modelsummary(request):
    return render (request,'modelsummary.html',{})

def userModelPredict(request):
    
    
    if request.method=='POST':
        user_model_path=r'News_Detector\saved_models\user_model.pkl'
        user_vector_path=r'News_Detector\saved_models\user_vectorized_weight.pkl'
        input_text=request.POST.get('text_input',' ')

    
        #print(input_text)
    
        with open(user_model_path,'rb') as file:
            model=pickle.load(file)
        with open(user_vector_path,'rb') as f:
            vectorize=pickle.load(f)
        #function for data cleaning and lemmatization to covert word to base word with actual meaning
        def lemmatize_func(text):
            text=text.lower()
            text=re.sub(r'[^a-zA-Z\s]','',text)
            tokenize_text=word_tokenize(text)
            filtered_token=[token for token in tokenize_text if token not in stopwords.words('english')]
            lemmatize_obj=WordNetLemmatizer()
            lemmatize_text=[lemmatize_obj.lemmatize(word) for word in filtered_token]
            cleaned_text=' '.join(lemmatize_text)
            return cleaned_text

        clean_data=lemmatize_func(input_text)

        listed_text=[clean_data]
            

        vectorized_text =vectorize.transform(listed_text)

        
        prediction_prob=model.predict_proba(vectorized_text)
        prediction=model.predict(vectorized_text)
        
        reliability=round(prediction_prob[0][1],2)
        print(f"The prediction is: {prediction}")
        print(f"Prediction type:{type(prediction)}")
        pr=None
        if(prediction[0])==1:
            print("The News is Reliable")
            pr='Reliable'
        else:
            print("Not reliable")
            pr='Not Reliable'


        all_data={'prediction': prediction[0],
                  'reliability':reliability,
                  'pr':pr}
        
    

    
        return render(request,'userModelPredict.html',all_data)
    else:

        return render(request,'userModelPredict.html',{})



