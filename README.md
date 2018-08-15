<h1>Facial Emotion Recognition</h1>

This repo consists of a very basic model for Facial Emotion Recognition. The dataset for Facial Emotion Recognition can be
downloaded from <a href= "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data">
here</a>.<br>
Now coming to the dataset, the data consists of 48x48 pixel grayscale images of faces. The faces have been automatically 
registered so that the face is more or less centered and occupies about the same amount of space in each image. 
The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories: <br>
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).<br>
train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6,
inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for 
each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the 
"pixels" column and our task is to predict the emotion column.

Once downloaded, we need to split the dataset in feature columns and label columns.
Over here I have splitted them to get 4 files: 
<ul>
<li>X_train.csv -> Contains the training data pixel values. 28709 examples</li>
<li>y_train.csv -> Contains a single column which is a numeric code ranging from 0 to 6, for the training purpose.</li>
<li>X_test.csv  -> Contains the testing data pixel values. 7178 examples</li>
<li>y_test.csv  -> Contains a single column which is a numeric code ranging from 0 to 6 for the testing purpose.</li>
</ul>

<h4>k-NN Classifier</h4>
The idea is very simple: instead of finding the single closest image in the training set, we will find the top 
<b>k</b> closest images, and have them vote on the label of the test image. In particular, when k = 1, we 
recover the Nearest Neighbor classifier. Intuitively, higher values of k have a smoothing effect that makes the classifier 
more resistant to outliers:<br>

![knnimage](https://user-images.githubusercontent.com/23143095/44161138-13c82a80-a0da-11e8-8458-b9493e9053cf.png)

In the implementation of kNN, I have used L2 distance Norm which has the geometric interpretation of computing the euclidean 
distance between two vectors. The distance takes the form:

![screenshot1](https://user-images.githubusercontent.com/23143095/44161497-22fba800-a0db-11e8-9050-9db847114ba0.png)
