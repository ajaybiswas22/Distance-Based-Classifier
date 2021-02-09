# Distance Based Classification
"""
@author: Ajay Biswas
220CS2184
M.Tech Information Security 
National Institute of Technology, Rourkela
"""

from sklearn import datasets
import statistics as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import sys
MAX = sys.maxsize

def MER_Error(X,Y):
  correct_count = 0
  for i in range(len(X)):
    if(X[i] == Y[i]):
      correct_count = correct_count + 1

  MER_val = 1 - (correct_count/len(X))
  return MER_val

def me(X,Y,d):
    error=0
    error_dist=0
    for i in range(len(X)):
        if(X[i] != Y[i]):
            error += 1
            error_dist+=d[i]
    m_error =error_dist/error
    return m_error

def euclidean_distance(p1,p2):
  
  distance = pow(sum([(a - b) ** 2 for a, b in zip(p1, p2)]),0.5)
  return distance 

def manhattan_distance(p1,p2):
    
  distance = 0
  for i in range(len(p1)):
      distance += abs(p1[i] - p2[i])
  return distance

def chessboard_distance(p1,p2):

  distance = abs(p1[0] - p2[0])
  for i in range(1,len(p1)):
      distance = max(distance,abs(p1[i] - p2[i]))
  return distance

def correlation_distance(p1,p2):
  norm_p1 = 0
  norm_p2 = 0
  
  for i in range(len(p1)):
      norm_p1 += (p1[i] - st.mean(p1))**2
      norm_p2 += (p2[i] - st.mean(p2))**2
      
  norm_p1 = norm_p1**0.5
  norm_p2 = norm_p2**0.5
  
  s = 0
  for i in range(len(p1)):
      s += (p1[i] - st.mean(p1))*(p2[i] - st.mean(p2))
  distance = 1 - s/(norm_p1*norm_p2)
  return distance


def minkowski_distance(p1,p2,p):
  
  s = 0
  for i in range(len(p1)):
      s += abs(p1[i] - p2[i])**p
  distance = s**(1/p)
  return distance


def cosine_distance(p1,p2):
  
  norm_p1 = 0
  norm_p2 = 0
  
  for i in range(len(p1)):
      norm_p1 += p1[i]**2
      norm_p2 += p2[i]**2
      
  norm_p1 = norm_p1**0.5
  norm_p2 = norm_p2**0.5
  
  s = 0
  for i in range(len(p1)):
      s += p1[i]*p2[i]
  distance = 1 - s/(norm_p1*norm_p2)
  return distance
  

def bray_curtis_distance(p1,p2):
  s1 = 0
  s2 = 0

  for i in range(len(p1)):
    s1 += abs(p1[i] - p2[i])
    s2 += abs(p1[i] + p2[i])
  
  distance = s1/s2
  return distance

def canberra_distance(p1,p2):
  distance = 0

  for i in range(len(p1)):
    s1 = abs(p1[i] - p2[i])
    s2 = abs(p1[i] + p2[i])
    distance += s1/s2

  return distance
 

def select_distance(p1,p2,distance_type):
# returns the calculated distance based on the type of distance provided
    
    if(distance_type == "euclidean"):
        return euclidean_distance(p1,p2)
    elif(distance_type == "manhattan" or distance_type == "cityblock"):
        return manhattan_distance(p1,p2)
    elif(distance_type == "chessboard" or distance_type == "chebyshev"):
        return chessboard_distance(p1,p2)
    elif(distance_type == "minkowski"):
        return minkowski_distance(p1,p2,3)
    elif(distance_type == "correlation"):
        return correlation_distance(p1,p2)
    elif(distance_type == "cosine"):
        return cosine_distance(p1,p2)
    elif(distance_type == "bray_curtis"):
        return bray_curtis_distance(p1,p2)
    elif(distance_type == "canberra"):
        return canberra_distance(p1,p2)
    else:
        return None

def distance_based_classifier(X,y,d_type,tp):
# X is a 2D matrix with two columns as features and rows as instances
# y is the true class labels
# d_type is the type of distance taken for classifying
# tp is the fraction of training dataset, the fractional testing dataset will be (1-tp)


    # Placing two features in two separate 2D arrays Species_x and Species_y
    # the rows of this 2D array determines each separate class
    cols = 1
    rows = 0
    for i in range(0,len(y)):
        if(y[i]!=rows):
            cols = 0
            rows = rows + 1
        cols = cols + 1;        
    rows = rows+1;
    
    Species_x = np.zeros((rows, cols))
    Species_y = np.zeros((rows, cols))
    
    cnt = 0
    for i in range(rows):
        for j in range(cols):
           Species_x[i][j] = X[cnt,0]
           Species_y[i][j] = X[cnt,1] 
           cnt = cnt + 1
           
    # rows = no. of classes
    # cols = no. of instances in each class
    
    ########################## Training Phase ########################## 
    
    # invalid training dataset size
    if(tp >= 1 or tp <= 0):
        return None
    
    percent = tp*100
          
    # Slicing from beginning
    train_range_s = 0
    train_range_e = int((cols/100)*percent)  
    
    all_centroid = np.zeros((rows, 2))
    
    # Taking mean of all points of each class and finding their centroid
    for k in range(rows):
        CL_x = Species_x[k][train_range_s:train_range_e]
        CL_mean_x = st.mean(CL_x)
        
        CL_y = Species_y[k][train_range_s:train_range_e]
        CL_mean_y = st.mean(CL_y)
        
        # (x,y) coordinates of centroid, column 0 - x, 1 - y
        all_centroid[k,0] = CL_mean_x;
        all_centroid[k,1] = CL_mean_y;
    
    
    ########################## Testing Phase ########################## 
    
    # Since we don't need class names now, we can simply merge all instances together
    
    # Slicing after the last training instance
    test_range_s = int((cols/100)*percent)  
    test_range_e = cols  
    
    C_x = np.zeros((rows, test_range_e - test_range_s))
    C_y = np.zeros((rows, test_range_e - test_range_s))
    for k in range(rows):
        C_x[k][:] = Species_x[k][test_range_s:test_range_e]
        C_y[k][:] = Species_y[k][test_range_s:test_range_e]       
    # Flattenning numpy array
    C_x = C_x.flatten()
    C_y = C_y.flatten()
    
    # predicted labels
    predicted = [0]*len(C_x);
    
    # actual labels
    # initially we keep labels in different rows w.r.t classes
    # later we will flatten the array
    actual = np.zeros((rows, test_range_e - test_range_s))
    beg = test_range_s
    end = cols
    for k in range(rows):
        actual[k][:] = y[beg:end]
        beg = beg + cols
        end = end + cols
    # flatten the array
    actual = actual.flatten()    
    
    # classifying points by measuring its distance from centroid of each class

    distances_predicted = [0]*len(C_x)

    min_dist = MAX
    for i in range (len(C_x)):
        for j in range(0,rows):
            
            distance = select_distance([all_centroid[j,0],all_centroid[j,1]],[C_x[i],C_y[i]],d_type)
            
            # invalid distance
            if(distance == None):
                return None
            
            # finding the minimum distance  
            if(min_dist > distance):
                min_dist = distance
                distances_predicted[i] = distance 
                lbl = j;
        
        # store predicted label
        predicted[i] = lbl;
        #reset min_dist
        min_dist = MAX

    # Calculating actual distances
    distances_actual = [0]*len(C_x);
    for i in range (len(C_x)):
      distances_actual[i] = select_distance([all_centroid[int(actual[i]),0],all_centroid[int(actual[i]),1]],[C_x[i],C_y[i]],d_type)
              
    # Accuracy Calculations
    mer_error = MER_Error(actual,predicted)

    mse_error = mse(distances_actual,distances_predicted)
    mae_error = mae(distances_actual,distances_predicted)
    mean_error = me(actual,predicted,distances_predicted)

    return [predicted, mer_error,mse_error,mae_error,mean_error]

######################################################################################

# iris dataset has three classes with 50 instances each
iris = datasets.load_iris()
X = iris.data[:, :2]  # first two features
y = iris.target    # contains true class labels 

euclidean = distance_based_classifier(X,y,"euclidean",0.6)
manhattan = distance_based_classifier(X,y,"manhattan",0.6)
chessboard = distance_based_classifier(X,y,"chessboard",0.6)
minkowski = distance_based_classifier(X,y,"minkowski",0.6)
cosine = distance_based_classifier(X,y,"cosine",0.6)
correlation = distance_based_classifier(X,y,"correlation",0.6)
chebyshev = distance_based_classifier(X,y,"chebyshev",0.6)
bray_curtis = distance_based_classifier(X,y,"bray_curtis",0.6)
canberra = distance_based_classifier(X,y,"canberra",0.6)
