import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def rms_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))



#The step function, calculates error, updates the gradients and returns new weights
def gradient_step(current_theta0, current_theta1, data, learning_rate):
    theta0_gradient = 0
    theta1_gradient = 0

    length = len(data)
    for i in range(length):
        x = data[i,0]
        y = data[i,1]

        theta0_gradient += (((current_theta1 * x) + current_theta0) - y)/length
        theta1_gradient += ((((current_theta1 * x) + current_theta0) - y)*x) / length

    new_theta0 = current_theta0 - (learning_rate * theta0_gradient)
    new_theta1 = current_theta1 - (learning_rate * theta1_gradient)

    return new_theta0, new_theta1



#The main linear regression function
def linear_regression(data, num_iterations, learning_rate):
    theta0 = 0.0
    theta1 = 0.0

    #run step function
    for i in range(num_iterations):
        theta0, theta1 = gradient_step(theta0, theta1, data, learning_rate)


    return theta0, theta1



#using the function
if __name__ == '__main__':

    data = np.genfromtxt("ex1data1.txt", delimiter="," , dtype=None)

    b, m = linear_regression(data, 10000, 0.01)
    print(b, m)

    print(rms_error(b,m, data))

    x = data[:,0]
    y = data[:,1]

    fig, ax = plt.subplots()

    ax.plot(x, (m *x) + b, color='red')  #plot line using the co efficient obtained from regression function
    ax.scatter(x,y)


    plt.show()
