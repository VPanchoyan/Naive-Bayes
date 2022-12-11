import numpy as np
import pandas as pd
import copy
class NaiveBayes:
  def __init__(self, smoothing=False):
      # initialize Laplace smoothing parameter
      self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
      # use this method to learn the model
      # if you feel it is easier to calculate priors 
      # and likelihoods at the same time
      # then feel free to change this method
      self.X_train = X_train
      self.y_train = y_train
      self.priors = self.calculate_priors()
      self.likelihoods = self.calculate_likelihoods()      
      
  def predict(self, X_test):
      # recall: posterior is P(label_i|feature_j)
      # hint: Posterior probability is a matrix of size 
      #       m*n (m samples and n labels)
      #       our prediction for each instance in data is the class that 
      #       has the highest posterior probability. 
      #       You do not need to normalize your posterior, 
      #       meaning that for classification, prior and likelihood are enough
      #       and there is no need to divide by evidence. Think why!
      # return: nd array of class labels (predicted)
      p = np.array([])
      prediction = np.array([])
      pred = []
      q = np.unique(self.y_train)
      for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
          for k in self.likelihoods.keys():
            pred.append(self.likelihoods[k][j][X_test[i][j]])
      for j in range(0,len(pred),24): 
        for i in range(len(self.likelihoods.keys())):
          p = np.append(p,self.priors[i]*np.prod(pred[j:j+24][i::4]))
      for i in range(0,len(p),4):
        prediction = np.append(prediction,q[p[i:i+4].argmax()])     
      return prediction 

  def calculate_priors(self):
      # recall: prior is P(label=l_i)
      # hint: store priors in a pandas Series or a list            
      priors = []
      labels = np.unique(self.y_train)
      for i in range(len(labels)):
        p = sum(self.y_train == labels[i])
        count = len(self.y_train)
        priors.append(p/count)       
      return priors
  
  def calculate_likelihoods(self):
      # recall: likelihood is P(feature=f_j|label=l_i)
      # hint: store likelihoods in a data structure like dictionary:
      #        feature_j = [likelihood_k]
      #        likelihoods = {label_i: [feature_j]}
      #       Where j implies iteration over features, and 
      #             k implies iteration over different values of feature j. 
      #       Also, i implies iteration over different values of label. 
      #       Likelihoods, is then a dictionary that maps different label 
      #       values to its corresponding likelihoods with respect to feature
      #       values (list of lists).
      #
      #       NB: The above pseudocode is for the purpose of understanding
      #           the logic, but it could also be implemented as it is.
      #           You are free to use any other data structure 
      #           or way that is convenient to you!
      #
      #       More Coding Hints: You are encouraged to use Numpy/Pandas as much as
      #       possible for all these parts as it comes with flexible and
      #       convenient indexing features which makes the task easier.
      likelihoods = {}
      features_values = []
      xt = self.X_train
      yt = self.y_train
      labels = np.unique(yt)
      for i in range(xt.shape[1]):
        features_values.append(np.unique(xt[:,i]))
      for j in labels:
        liks = []
        for i in range(xt.shape[1]):
          likelihood = []
          for k in range(len(features_values[i])):  # k = 0
            s = sum((xt[:,i] == features_values[i][k]) & (yt == j))
            p = sum(yt == j)
            likelihood.append((s+1)/(p+len(features_values[i])))
          liks.append(likelihood)
        likelihoods[j] = liks
      result = {}
      for j in labels:
        result[j] = {}
        for i in range(len(likelihoods[j])):
          result[j][i] = dict(zip(features_values[i],likelihoods[j][i]))
      return result
