'''

import math

import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iteration = 1000
    n = len(x)
    cost = 0
    learning_rate = 0.001

    for i in range(iteration):
        y_predicted = m_curr*x + b_curr
        cost_previous = cost
        cost = (1/n) *sum([val**2 for val in (y - y_predicted)])#loss function nkal ra hy

        m_derivative = -(2/n)*sum(x*(y-y_predicted))#derivative of slope/coef_
        b_derivative = -(2/n)*sum(x*(y-y_predicted))#derivaive of intercept

        b_curr = b_curr - (learning_rate * b_derivative)#update b using gradient descent formula
        m_curr = m_curr - (learning_rate *m_derivative)


        # this condition check if cost is same as previous iteration then stop the loop and print cost that will be min value of function
        if(math.isclose(cost_previous , cost, rel_tol = 1e-20)):
            print("THE LOCAL MINIMIM VALUE IS :",cost)
            print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))
            break


x = np.array([1,2,3,4,5,6])#FEATURES
y = np.array([5,7,9,11,13,15])#LABLES

gradient_descent(x,y)


'''