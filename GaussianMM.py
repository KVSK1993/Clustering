# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:38:37 2017

@author: Karan Vijay Singh
"""

import timeit
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#starting time of program
startTime = timeit.default_timer()

#Reading of training data
training_data = pd.read_csv('MNIST_X_train.csv',header=None,nrows=10000).as_matrix()
training_result = pd.read_csv('MNIST_y_train.csv',header=None,nrows=10000).as_matrix()

#Training Data Dimensions
dataPts=training_data.shape[0]
dims=training_data.shape[1]

#reading of test data
testing_data = pd.read_csv('MNIST_X_test.csv',header=None).as_matrix()
testing_result = pd.read_csv('MNIST_y_test.csv',header=None).as_matrix()

#Test Data Dimensions
testDataPts=testing_data.shape[0]
testDims=testing_data.shape[1]

#TAKING DOT PRODUCT OF DATA POINTS OF TEST DATA USED IN TESTING
xTestDotx = np.zeros((testDataPts,1),dtype = float)
for i in range(testDataPts):
    xTestDotx[i] = np.dot(testing_data[i],testing_data[i])

noOfIterations=500
tolerance = 10**-5
k = 5 #no of models
NumberofClasses = 10

training_dataC0=None
xDotxC0=None
training_dataC1=None
xDotxC1=None
training_dataC2=None
xDotxC2=None
training_dataC3=None
xDotxC3=None
training_dataC4=None
xDotxC4=None
training_dataC5=None
xDotxC5=None
training_dataC6=None
xDotxC6=None
training_dataC7=None
xDotxC7=None
training_dataC8=None
xDotxC8=None
training_dataC9=None
xDotxC9=None
perClassDataVector = np.zeros((10,1),dtype = float)
#####################################################################################
#SPLITTING THE DATA INTO SEPARATE CLASSES FOR TRAINING
#########################################################################################

for a in range(training_data.shape[0]):
    xDataPt = np.reshape(np.copy(training_data[a]),(1, dims))
    xDotx = np.reshape(np.dot(xDataPt,np.transpose(xDataPt)),(1,1))
    #Separating  data of class 0
    if training_result[a] == 0:
        perClassDataVector[0]+=1
        if training_dataC0 is None:
            xDotxC0 = xDotx
            training_dataC0 = xDataPt
        else:
            xDotxC0 = np.concatenate((xDotxC0,xDotx),axis = 0)
            training_dataC0 = np.concatenate((training_dataC0,xDataPt), axis = 0)
    #Separating  data of class 1                   
    if training_result[a] == 1:
        perClassDataVector[1]+=1
        if training_dataC1 is None:
            xDotxC1 = xDotx
            training_dataC1 = xDataPt            
        else:
            xDotxC1 = np.concatenate((xDotxC1,xDotx),axis=0)
            training_dataC1 = np.concatenate((training_dataC1,xDataPt), axis = 0)            
    #Separating  data of class 2       
    if training_result[a] == 2:
        perClassDataVector[2]+=1
        if training_dataC2 is None:
            xDotxC2 = xDotx
            training_dataC2 = xDataPt            
        else:
            training_dataC2 = np.concatenate((training_dataC2,xDataPt), axis = 0)
            xDotxC2 = np.concatenate((xDotxC2,xDotx),axis=0)
    #Separating  data of class 3        
    if training_result[a] == 3:
        perClassDataVector[3]+=1
        if training_dataC3 is None:
            xDotxC3 = xDotx
            training_dataC3 = xDataPt            
        else:
            xDotxC3 = np.concatenate((xDotxC3,xDotx),axis=0)
            training_dataC3 = np.concatenate((training_dataC3,xDataPt), axis = 0)
    #Separating  data of class 4                 
    if training_result[a] == 4:
        perClassDataVector[4]+=1
        if training_dataC4 is None:
            xDotxC4 = xDotx
            training_dataC4 = xDataPt           
        else:
            xDotxC4 = np.concatenate((xDotxC4,xDotx),axis=0)
            training_dataC4 = np.concatenate((training_dataC4,xDataPt), axis = 0)   
    #Separating  data of class 5        
    if training_result[a] == 5:
        perClassDataVector[5]+=1
        if training_dataC5 is None:
            xDotxC5 = xDotx
            training_dataC5 = xDataPt            
        else:
            xDotxC5 = np.concatenate((xDotxC5,xDotx),axis=0)
            training_dataC5 = np.concatenate((training_dataC5,xDataPt), axis = 0)        
    #Separating  data of class 6        
    if training_result[a] == 6:
        perClassDataVector[6]+=1
        if training_dataC6 is None:
            xDotxC6 = xDotx
            training_dataC6 = xDataPt       
        else:
            xDotxC6 = np.concatenate((xDotxC6,xDotx),axis=0)
            training_dataC6 = np.concatenate((training_dataC6,xDataPt), axis = 0)    
    #Separating  data of class 7
    if training_result[a] == 7:
        perClassDataVector[7]+=1
        if training_dataC7 is None:
            xDotxC7 = xDotx
            training_dataC7 = xDataPt            
        else:
            xDotxC7 = np.concatenate((xDotxC7,xDotx),axis=0)
            training_dataC7 = np.concatenate((training_dataC7,xDataPt), axis = 0)
    #Separating  data of class 8                  
    if training_result[a] == 8:
        perClassDataVector[8]+=1
        if training_dataC8 is None:
            xDotxC8 = xDotx
            training_dataC8 = xDataPt            
        else:
            xDotxC8 = np.concatenate((xDotxC8,xDotx),axis=0)
            training_dataC8 = np.concatenate((training_dataC8,xDataPt), axis = 0)           
    #Separating  data of class 9        
    if training_result[a] == 9:
        perClassDataVector[9]+=1
        if training_dataC9 is None:
            xDotxC9 = xDotx
            training_dataC9 = xDataPt    
        else:
            xDotxC9 = np.concatenate((xDotxC9,xDotx),axis=0)
            training_dataC9 = np.concatenate((training_dataC9,xDataPt), axis = 0)
            
################################################################################            
perClassDataVector = perClassDataVector/dataPts 

noOfModels=5
gauss = noOfModels*NumberofClasses

#function to make GAUSSIAN MIXTURES MODEL FOR DIFFERENT CLASSES
def gaussian_misture_model(noOfModels,training_input,xDotx): 
    dataPt = training_input.shape[0]
    dim = training_input.shape[1]
    pieArray = np.random.rand(noOfModels,1)
    MiuMatrix = np.random.rand(dim,noOfModels) 
    CovArray = np.random.rand(noOfModels,1)+10 # shifting
    
    logLikeliCurr = 0.0
    for i in range(noOfIterations):
        rikMatrix = None
        loglikeliOld = logLikeliCurr
        for k in range(noOfModels):
            miuArrayK = np.reshape(MiuMatrix[:,k],(dim,1))
            kCovariance = CovArray[k]
            pieK = pieArray[k] 
            miuTmiu = np.dot(np.transpose(miuArrayK),miuArrayK)
            xMiu = np.dot(training_input,miuArrayK)
            xNorm = xDotx + miuTmiu -2*xMiu 
            xNorm = (-1/(2*kCovariance))*xNorm
            xNorm = np.log(pieK)-(dim/2)*np.log(kCovariance)+xNorm
            #creating rik matrix 
            if k == 0:
                rikMatrix = xNorm
            else:
                rikMatrix = np.concatenate((rikMatrix,xNorm),axis = 1)
                
        if rikMatrix is not None:
            #print(i)
            rikMatrix=rikMatrix*1./np.max(rikMatrix, axis=0)
            riArray = np.reshape(np.sum(rikMatrix,axis=1),(dataPt,1))
            rikMatrix = rikMatrix/riArray 
            logLikeliCurr = -np.sum(riArray,axis=0) 
            #comparing the log likelihood values to break the loop
            if i>1 and abs(logLikeliCurr-loglikeliOld)<=np.abs(logLikeliCurr)*tolerance:
                break
            sumR = np.sum(rikMatrix,axis = 0)
            postrSumK = np.reshape(sumR, (sumR.shape[0],1)) #Posterior
            pieArray = postrSumK/dataPt 
            
            MiuMatrix = None
            for j in range(noOfModels):
                colK = rikMatrix[:,j]
                colK = np.reshape(colK,(colK.shape[0],1))   
                #Calculating miu array 
                miuKMatrix = colK*training_input
                miuKArray = (np.sum(miuKMatrix,axis=0))/postrSumK[j]  
                miuKArray = np.reshape(miuKArray,(miuKArray.shape[0],1)) 
                
                if j ==0:
                    MiuMatrix = miuKArray
                else:
                    MiuMatrix = np.concatenate((MiuMatrix,miuKArray), axis = 1) 
            
                miuTmiu = np.dot(np.transpose(miuKArray), miuKArray) 
                xMiu = np.dot(training_input,miuKArray)
                xNorm = xDotx + miuTmiu -2*xMiu
                xNorm = xNorm*colK
                covK = (np.sum(xNorm,axis=0))/(dim*postrSumK[j]) 
                CovArray[j] = covK
                
    return CovArray,MiuMatrix,pieArray


#######################################################################
# Calculating gmm for each class by calling the function 
####################################################################

finalMius=None
finalCovariance=None
finalPieK = None

#evaluating covariance, mean and piek value for class 0
covC0,miuC0,pieC0 = gaussian_misture_model(noOfModels, training_dataC0, xDotxC0)
finalCovariance,finalMius ,finalPieK=covC0,miuC0,pieC0
#evaluating covariance, mean and piek value for class 1
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC1, xDotxC1)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 2
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC2, xDotxC2)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 3
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC3, xDotxC3)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 4
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC4, xDotxC4)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 5
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC5, xDotxC5)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 6
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC6,xDotxC6)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 7
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC7, xDotxC7)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 8
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC8, xDotxC8)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix
#evaluating covariance, mean and piek value for class 9
covC1,miuC1,pieC1 = gaussian_misture_model(noOfModels, training_dataC9, xDotxC9)
finalCovariance = np.concatenate((finalCovariance,covC1),axis = 0) # k=5 values
finalMius = np.concatenate((finalMius,miuC1), axis = 1) #matrix
finalPieK = np.concatenate((finalPieK,pieC1),axis = 0) #matrix

###############################################################
#for testing 
###################################################################

probMatrix = None

#vectorising the test data and creating probablity matrix for each class having k no of gaussians
for j in range(gauss):
    miuArray = np.reshape((finalMius[:,j]),(testDims,1))
    Covariance = finalCovariance[j]
    miuTmiu = np.dot(np.transpose(miuArray), miuArray)
    Xmiu = np.dot(testing_data,miuArray)
    xNorm = xTestDotx + miuTmiu - 2*Xmiu 
    xNorm = (-1/(2*Covariance))*xNorm
    xNorm = np.log(finalPieK[j])-(testDims/2)*np.log(Covariance)+xNorm
    kClass = int(j/noOfModels)
    xNorm = perClassDataVector[kClass]*xNorm
    #creatingthe matrix whoich contains the probabilities
    if j == 0:
        probMatrix = xNorm 
    else:
        probMatrix = np.concatenate((probMatrix,xNorm),axis = 1)
        

finalMatrix = np.empty((testDataPts,10))
for i in range(NumberofClasses):
    finalMatrix[:,i] =  np.sum(probMatrix[:,noOfModels*i:noOfModels*(i+1)],axis=1)    

#predicting class for each test data point by finding max probabilty index which is the class number
predictedClasses = np.argmax(finalMatrix,axis = 1)
predictedClasses = np.reshape((predictedClasses),(testDataPts,1)) # predicted class of each data point
#print("predictedClasses", predictedClasses)
#calculating test error rate
error = ((testing_result-predictedClasses)!=0).sum()
print("Error percent for k=%d is %f" % (noOfModels,(error*100)/testDataPts))

stopTime = timeit.default_timer()
print ("TotalTime to run the code ", stopTime - startTime)
