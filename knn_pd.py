import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import sys


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

  
ml_algo = str(sys.argv[1]).upper()
dataset = str(sys.argv[2])


class MLChoice:

  def __init__(self, learning_rate=0.001 , lambda_parameter = 0.01 , iter=1000 ):
    self.k=3

    
    self.ML = ml_algo
    # self.dataset = "data_banknote_authentication.txt"
    self.dataset= dataset
    self.X_train= None
    self.y_train = None
    self.X_test = None
    self.y_test = None
    self.y_pred= []
    self.unique_values = None #Stores unique values for the output. Usually a string (Red , Blue),etc
    self.lr=learning_rate
    self.lambda_parameter=lambda_parameter
    self.iter = iter
     
    self.w=0
    self.b=0

    




  
  def get_accuracy(self):

    with open('output.txt' , 'w') as file:
      file.write(("Dataset: " +  self.dataset))
      file.write(("Model used: " + self.ML))


    
    
  
    if self.ML == "KNN" :

        with open('output.txt' , 'a') as file:
          

            print("\nAccuracy of Training(Scratch): ", accuracy_score(self.y_pred , self.y_test) , file=file)

        
            clf = KNeighborsClassifier(n_neighbors=self.k)
            clf.fit(self.X_train , self.y_train)

            clf_y_pred = clf.predict(self.X_test)

            print("Accuracy of model: " , accuracy_score(clf_y_pred, self.y_test) , file =file)
    else:
      
      #Get the slef_pred values
      self.y_pred = self.svm_predict()
      
      with open('output.txt' , 'a') as file:

        print("\nAccuracy of model: " , accuracy_score(self.y_pred , self.y_test), file=file)

        
        svm = SVC(kernel='linear')
        svm.fit(self.X_train , self.y_train)
        svm_y_pred = svm.predict(self.X_test)
        print("Accuracy of model: " , accuracy_score(svm_y_pred, self.y_test),file=file)

      



    
  def load_data(self):
    data=pd.read_csv(self.dataset)

    # Preprocess data

    # Sometimes the output class is not in integer form (0,1) but a string. Convert it to zero and one and store it somewhere
    unique_values = data.iloc[:,-1].unique()
    data.iloc[:,-1] = data.iloc[:,-1].replace({unique_values[0]:0 , unique_values[1]:1})
      

    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    #Train , test , split
    self.X_train ,  self.X_test , self.y_train , self.y_test = train_test_split(X,y,test_size=0.3 , random_state=42)
    
    # print("Succesfully loaded the dataset")
    # print(self.X_train)
    # print(self.X_test)
    
  def predict(self):
    if self.ML=="KNN":
      self.knn_predict()
    else:
      self.svm_fit()
      self.svm_predict()

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
     

      y_lables.append(most_common(labels))


      
      
    self.y_pred =  y_lables
    # print("Output label")
    # print(self.y_pred)
    # print("True label")
    # print(self.y_test)

  def svm_fit(self):
    #We are passing the X_train and Y_train
    n_samples , n_features = self.X_train.shape

    #Map values to zero
    y_ = np.where(self.y_train<=0,-1,1)

    self.w=np.zeros(n_features)
    self.b=0
    
    # Perform the gradient descent
    for i in range(self.iter):
      # For every value in the training X
      #Check if the condition is met
      for index , x_i in enumerate(self.X_test):
        constraint= y_[index] * (np.dot(x_i , self.w)-self.b)>=1
        if constraint:
          # Update the w value
          self.w -= self.lr * (2*self.lambda_parameter*self.w)
        else:
          # Update both values
           self.w -= self.lr * (2 * self.lambda_parameter * self.w - np.dot(x_i, y_[index]))
           self.b -= self.lr * y_[index]

    # At this point , we have values for w, b and we can perform prediction
    print("I fit the data")
      


  def svm_predict(self):
    value = np.dot(self.X_test , self.w)-self.b
    return np.sign(value)
 
knn = MLChoice()
knn.load_data()
knn.predict()
knn.get_accuracy()
