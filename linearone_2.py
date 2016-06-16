from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour


#Evaluate the linear regression

#computing the cost to monitor the convergence
def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training data 
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


#get the training values from trainingdata.txt
data = loadtxt('trainingdata.txt', delimiter=',')

#Plot the graph
scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Linear Regression- One variable using Batch Gradient descent')
xlabel('X training Example')
ylabel('y training examples')
#show()

X = data[:, 0]
y = data[:, 1]


#number of y training samples
m = y.size

#Add a column of ones to X (interception data)
it = ones(shape=(m, 2))
it[:, 1] = X

#Initialize theta parameters
theta = zeros(shape=(2, 1))

#Some gradient descent settings
iterations = 1500
alpha = 0.01

#compute and display initial cost
print' value of cost j(theta)'
print compute_cost(it, y, theta)

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
print'values of  theta_0 and theta_1 repectively'
print theta

print 'using equation y= theta_0 + theta_1*x predict the values'
#Predict values 

xString = input("Enter a x1: ")
x = int(xString)
x2String = input("Enter a x2: ")
x2= int(x2String)
predict1 = array([1, x]).dot(theta).flatten()
print 'For x1, y is %f' % (predict1 * 1)
predict2 = array([1, x2]).dot(theta).flatten()
print 'For x2 , y is %f' %(predict2* 1)

#Plot the results
result = it.dot(theta).flatten()
plot(data[:, 0], result)
show()


#Grid over which we will calculate J
#theta0_vals = linspace(-10, 10, 100)
#theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
#J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fill out J_vals
#for t1, element in enumerate(theta0_vals):
 #   for t2, element2 in enumerate(theta1_vals):
  #      thetaT = zeros(shape=(2, 1))
   #     thetaT[0][0] = element
    #    thetaT[1][0] = element2
     #   J_vals[t1, t2] = compute_cost(it, y, thetaT)

#Contour plot
#J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
#contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
#xlabel('theta_0')
#ylabel('theta_1')
#scatter(theta[0][0], theta[1][0])
#show()
