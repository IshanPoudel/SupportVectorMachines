import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def euclidean(p1,p2):
    
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist



class MLChoice:

  def __init__(self ):
    self.k=3

    
    self.ML = "KNN"
    # self.dataset = "data_banknote_authentication.txt"
    self.dataset="sonar.txt"
    self.X_train= None
    self.y_train = None
    self.X_test = None
    self.y_test = None
    self.y_pred= []

    



  
  
  def get_accuracy(self):

   #Based on y_pred and y_predict find the accruacy
    print(accuracy_score(self.y_pred , self.y_test))

    # use a skleanr model. 
    clf = KNeighborsClassifier(n_neighbors=self.k)
    clf.fit(self.X_train , self.y_train)

    clf_y_pred = clf.predict(self.X_test)

    print("Accuracy of model: " , clf.score(self.X_test, self.y_test))



    
  
  def load_data(self):
    data=pd.read_csv(self.dataset)
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    #Train , test , split
    self.X_train ,  self.X_test , self.y_train , self.y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
    
    # print("Succesfully loaded the dataset")
    # print(self.X_train)
    # print(self.X_test)
    


  def predict( self ):
    
    # Returns the values of the y values.
    y_lables=[]

    # For each item in the test_state , 
    # find the euclidean distance and find the class
    #Return all the y predictions

    for item in self.X_test:
      # Sort all point distace for the item and training set
      point_dist=[]

      for j in range(len(self.X_train)):
        point_dist.append(euclidean(item , self.X_train[j,:]))
      

      #Once you have all the euclidean distance , turn it into an np array
      point_dist = np.array(point_dist)

    #   print(item)
    #   print("Here are the distacne")
    #   print(point_dist)

      #Sort the array
      #Arg sort returns --the index--
      point_dist = np.argsort(point_dist)
      point_dist = point_dist[:self.k]




      
      labels = self.y_train[point_dist]
    #   print("Lables sorted by the distance")
    #   print(labels)

      #You get all the labels. 

      frequent = mode(labels)
      frequent = frequent.mode[0]
      y_lables.append(frequent)


      
      # Once sorted , need to pick the label that occurs the 
      # most and add it to lables.
    self.y_pred =  y_lables
    # print("Output label")
    # print(self.y_pred)
    # print("True label")
    # print(self.y_test)

 
knn = MLChoice()
knn.load_data()
knn.predict()
knn.get_accuracy()
