# News_Detector
News_Detector is a simple Django-based app that uses an ML algorithm to determine whether a news text is reliable or not.
This app has used logistic regression for initial deployment and will use more powerful algorithms in the near future. So it will be updated with the progress of time.

The packages, library, and other dependencies needed for this app are listed in the requirements.txt file.


To train the model a dataset is used from Kaggle (link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). This dataset has more than 70 thousand rows. 

All business logics are written in a views.py file that is kept in the News_Detector folder.

This website has a total 6 pages such as HOME, about, contact, ModelSummary, NewsDetect, and UserModelPrediction.
On the HOME page, a user can check the news using a pre-trained model trained on the main large dataset. The About and Contact page contains the about and contact information of the website. The NewsDetect page gives the facility to the users to train an ML model based on their dataset in CSV format. But this dataset must contain a title, text, and label column. After submitting the dataset user has to wait a certain time to train the model, after training a new model will be saved into a saved model folder contained within the News_Detector folder having the first word of the name "user". Remember every time a user uploads a new dataset and submits, a new model will be trained and the previous one will be removed. Only after completing the training of the new model based on the user data, the user can check a news text based on their new model from the UserModlPrediction page. After completion of the training, the user will see a summary of their newly trained model in the News_Detect page containing the train test dataset confusion matrix and ROC curves.
In the ModelSummary page, there is a total summary of the main pre-trained model used in the HOME page. This summary contains the Confusion matrix and ROC curve of the train, test dataset of the main pre-trained model.

Note: When shifting from one page to another, remember to first back to the homepage and then to your desired page, otherwise the URL will not be able to recognize the path of the target page. When you will try to train a model based on your own data, make sure the data is small otherwise it will take a long time and sometimes may not work.
