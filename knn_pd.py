import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def euclidean(p1,p2):
    
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

#Helper function for getting the most common element

def most_common(num):
  count = {}

  for a in num:
    if a in count:
      count[a]+=1
    else:
      count[a]=1

  max_value = max(count , key=count.get)

  return max_value

  


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
    print("Accuracy of Training(Scratch): ", accuracy_score(self.y_pred , self.y_test))

    # use a skleanr model.
    # 
    if self.ML == "KNN":
     
        clf = KNeighborsClassifier(n_neighbors=self.k)
        clf.fit(self.X_train , self.y_train)

        clf_y_pred = clf.predict(self.X_test)

        print("Accuracy of model: " , accuracy_score(clf_y_pred, self.y_test))
    else:
      svm = SVC(kernel='linear')
      svm.fit(self.X_train , self.y_train)
      svm_y_pred = svm.predict(self.X_test)
      print("Accuracy of model: " , accuracy_score(svm_y_pred, self.y_test))

      



    
  
  def load_data(self):
    data=pd.read_csv(self.dataset)
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    #Train , test , split
    self.X_train ,  self.X_test , self.y_train , self.y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
    
    # print("Succesfully loaded the dataset")
    # print(self.X_train)
    # print(self.X_test)
    
  def predict(self):
    if self.ML=="KNN":
      self.knn_predict()
    else:
      return

  def knn_predict( self ):
    
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

      #Get the most frequent lable


    #   frequent = mode(labels)
    #   frequent = frequent.mode[0]
    #   y_lables.append(statistics.mode(labels))
      y_lables.append(most_common(labels))


      
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
