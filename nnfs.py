# Creating a spiral dataset 
import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(points, noise = 0.5):
    n = points // 3 # Divide points eqally among three spirals
    theta = np.sqrt(np.random.rand(n))*2*np.pi # Angle of spiral
    # Generate the first spiral
    x1 = np.cos(theta)*theta
    y1 = np.sin(theta)*theta

    x1 += np.random.normal(scale=noise, size=n)
    y1 += np.random.normal(scale=noise, size=n)

    # Generate the second spiral
    x2 = -np.cos(theta)*theta
    y2 = -np.sin(theta)*theta

    x2 += np.random.normal(scale=noise,size=n)
    y2 += np.random.normal(scale=noise,size=n)
    
    # Generate the third spiral
    x3 = np.cos(theta + np.pi/1.7)*theta
    y3 = np.sin(theta + np.pi/1.7)*theta

    x3 += np.random.normal(scale=noise, size=n)
    y3 += np.random.normal(scale=noise, size=n)

    # Combine the points and create labels
    X = np.vstack((np.vstack((x1,y1)).T,
               np.vstack((x2,y2)).T,
               np.vstack((x3,y3)).T))
    Y = np.hstack((
    np.zeros(n, dtype='int'), 
    np.ones(n, dtype='int'), 
    np.full(n, 2, dtype='int')))

    return X, Y

# Generate the data
points = 900
noise = 0.2
X, Y = generate_spiral_data(points, noise)

# plot the data
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Spectral)
plt.title("3D Spiral Data With Noise")
plt.show()


# Neural Network From Scratch
class NN:
    def __init__(self, n_features, n_hidden, n_classes):
        self.n = n_features
        self.h = n_hidden
        self.c = n_classes
        self.W1 = 0.01*np.random.randn(self.n, self.h)
        self.b1 = np.zeros((1,self.h))
        self.W2 = 0.01*np.random.randn(self.h, self.c)
        self.b2 = np.zeros((1,self.c))

    def fwd_prop(self,X):
        Z1 = np.dot(X, self.W1)+self.b1
        A1 = np.maximum(0,Z1)
        Z2 = np.dot(A1,self.W2)+self.b2
        Z2 = np.exp(Z2)
        A2 = Z2/np.sum(Z2, axis=1, keepdims=True)

        return A1, A2
    
    def c_loss(self, Y, prob):
        num_example = Y.shape[0]
        correct_log_prob = -np.log(prob[range(num_example),Y])
        loss = np.sum(correct_log_prob)/num_example

        return loss
    
    def back_prop(self,X,A1,A2,Y):
        # Compute the gradient on scores
        num_example = Y.shape[0]
        dZ2 = A2
        dZ2[range(num_example),Y]-=1
        dZ2/=num_example

        # First backpop into parameters W2 and b2
        dW2 = np.dot(A1.T,dZ2)
        db2 = np.sum(dZ2,axis=0,keepdims=True)

        # First backprop into hidden layers A1
        dA1 = np.dot(dZ2,self.W2.T)
        # Backprop the ReLU non-linearity
        dA1[A1<=0]=0
        # Finally in W, b
        dZ1 = dA1
        dW1 = np.dot(X.T,dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
       
        return dW1,db1,dW2,db2
        



    def fit(self, X, Y, lr, reg, max_iters):
	    
        num_examples = X.shape[0]
        for i in range (max_iters):
            # Forward prop 
            A1, A2 = self.fwd_prop(X)
            # Calculate loss
            data_loss = self.c_loss(Y,A2)
            reg_loss = 0.5*reg*np.sum(self.W1*self.W1)+0.5*reg*np.sum(self.W2*self.W2)
            loss = data_loss + reg_loss
            if i%1000 == 0:
                print("iteration % d:loss % f " %(i, loss))
            
            dW1,db1,dW2,db2 = self.back_prop(X,A1,A2,Y)

            # Add reg gradient distribution 
            dW2 += reg*self.W2
            dW1 += reg*self.W1

            # Prepare a parameter update
            self.W1 -= lr*dW1
            self.b1 -= lr*db1

            self.W2 -= lr*dW2
            self.b2 -= lr*db2

    def predict(self, X):
        A1 = np.maximum(0,np.dot(X,self.W1)+self.b1)
        Z2 = np.dot(A1,self.W2)+(self.b2)
        y_hat = np.argmax(Z2,axis=1) # Taking index of Maximum Probability

        return y_hat
    
nn_model = NN(n_features=2, n_hidden=100, n_classes=3)
nn_model.fit(X,Y,lr=1,reg=1e-3,max_iters=10000)

Z = nn_model.predict(X)
unique_values, counts = np.unique(Z, return_counts=True)
print(("Unique Values:", unique_values))
print("counts:", counts)

# Plotting the prediction 
# Create a meshgrid
# --- Fixed Plotting Section ---

h = 0.02 # step size
# Use X (features) to define boundaries, not Y (labels)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict across the entire meshgrid
# Use the model's predict method
Z = nn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title("Decision Boundary of NN")
plt.show()
