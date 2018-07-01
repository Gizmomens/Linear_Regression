import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_price(thetas, features):

    no_of_terms = len(thetas)
    price = 0
    for i in range(no_of_terms):
        price += features[i] * thetas[i]

    return price


def plot_graph(costs):

    x=[]

    #put no of itrs in x
    for i in range(len(costs)):
        x.append(i+1)

    plt.plot(x,costs)
    plt.show()


def mean_normalization(data):


    length = len(data)
    no_of_features = len(data[0]) - 1   #last column is target value
    sums = data.sum(axis=0)          #sums each column of data and puts the sum in "sum"
    
    max =[]
    min =[]
    mean=[]
    
    for i in range(no_of_features):

        max.append(data[:,i].max())    #iteratively get max, min ans mean of each column/feature
        min.append(data[:,i].min())
        mean.append(sums[i]/length)

        for j in range (length):
            data[j, i] = (data[j, i] - mean[i])/(max[i]-min[i])




    return mean, max, min


def rms_error(weights, data):

    col = np.ones((len(data), 1), dtype=float)  # gen column of 1's with as many rows as data
    data = np.hstack((col, data))

    total_error = 0
    length = len(data)
    no_of_weights = len(weights)

    for i in range(length):

        y = data[i,no_of_weights]
        pred_val = 0
        for j in range(no_of_weights):
            pred_val += weights[j] * data[i,j]

        total_error += (y - pred_val) ** 2

    return total_error/length


def gradient_step(current_thetas, data, learning_rate):

    gradients = []
    new_thetas = []
    no_of_gradients = len(data[0]) - 1
    length = len(data)
    total_error = 0

    #init gradients to 0
    for i in range(no_of_gradients):
        gradients.append(0)


    for i in range(length):
        for j in range(no_of_gradients):

            x = data[i, j]
            y = data[i, no_of_gradients]

            pred_val = 0

            #calc predicted val
            for k in range(no_of_gradients):
                pred_val += current_thetas[k] * data[i,k]

            gradients[j] += (((pred_val) - y)*x) / length
            total_error += (y - pred_val) ** 2            #calculate cost at each itr for the plot

    for j in range(no_of_gradients):
        new_thetas.append(current_thetas[j] - (learning_rate * gradients[j]))

    cost = total_error/length
    return new_thetas, cost

'''
method = method of calculating best fit, i.e either gradient descent or Matrix multiplication
function assumes last column of 'data' will be target variable and rest will be features
in case of matrix, num_iterations and learning rate will be ignored
'''

def muiltivar_linear_regression(data, num_iterations, learning_rate, method):

    costs = [0]*num_iterations                         #init list with num_itr 0s

    #add row of 1s to start of data set for dummy x0 vars
    col = np.ones((len(data),1), dtype=float)         #gen column of 1's with as many rows as data
    data = np.hstack((col,data))                    #combine the two np arrays

    if(method == "itr"):

        #init weights
        thetas = []
        for i in range(len(data[0])-1):         #last column is target
            thetas.append(0)

        #run step fucntion
        for i in range(num_iterations):
            thetas, costs[i] = gradient_step(thetas, data, learning_rate)

        #plot graph showing decrease in cost func with each itr
        plot_graph(costs)
        return thetas

    elif(method == "matrix"):

        no_of_features = len(data[0])-1
        X = data[:,0:no_of_features]      #get feature matrix X
        Y = data[:,no_of_features:no_of_features+1]

        X_trans = np.transpose(X)
        thetas = np.linalg.pinv(X_trans.dot(X)).dot(X_trans.dot(Y))

        return thetas












