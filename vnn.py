# %% 1 
# Package imports 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
V, o = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(V[:,0], V[:,1], s=40, c=o,cmap=plt.cm.Spectral)
plt.show()

# Train the logistic rgeression classifier 
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(V, o)

# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below. 
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding 
    x_min, x_max = V[:, 0].min() - .5, V[:, 0].max() + .5
    y_min, y_max = V[:, 1].min() - .5, V[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(V[:, 0], V[:, 1], c=o, cmap=plt.cm.Spectral)

# Plot the decision boundary 
plot_decision_boundary(lambda v: clf.predict(v))
plt.title("Logistic Regression")
plt.show()

num_examples = len(V) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    U1, c1, U2, c2 = model['U1'], model['c1'], model['U2'], model['c2']
    # Forward propagation to calculate our predictions
    z1 = V.dot(U1) + c1
    d1 = np.tanh(z1)
    z2 = d1.dot(U2) + c2
    exp_scores = np.exp(z2)
    ohat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(ohat[range(num_examples), o])
    Loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    Loss += reg_lambda/2 * (np.sum(np.square(U1)) + np.sum(np.square(U2)))
    return 1./num_examples * Loss

# Helper function to predict an output (0 or 1)
def predict(model, v):
    U1, c1, U2, c2 = model['U1'], model['c1'], model['U2'], model['c2']
    # Forward propagation
    z1 = v.dot(U1) + c1
    d1 = np.tanh(z1)
    z2 = d1.dot(U2) + c2
    exp_scores = np.exp(z2)
    ohat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(ohat, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    U1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    c1 = np.zeros((1, nn_hdim))
    U2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    c2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = V.dot(U1) + c1
        d1 = np.tanh(z1)
        z2 = d1.dot(U2) + c2
        exp_scores = np.exp(z2)
        ohat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Backpropagation
        grad_z2 = ohat
        grad_z2[range(num_examples), o] -= 1
        grad_U2 = (d1.T).dot(grad_z2)
        grad_c2 = np.sum(grad_z2, axis=0, keepdims=True)
        grad_d1 = grad_z2.dot(U2.T)
        grad_z1 = grad_d1 * (1 - np.power(d1, 2))
        grad_U1 = np.dot(V.T, grad_z1)
        grad_c1 = np.sum(grad_z1, axis=0)

        # Add regularization terms (c1 and c2 don't have regularization terms)
        grad_U2 += reg_lambda * U2
        grad_U1 += reg_lambda * U1

        # Gradient descent parameter update
        U1 += -epsilon * grad_U1
        c1 += -epsilon * grad_c1
        U2 += -epsilon * grad_U2
        c2 += -epsilon * grad_c2

        # Assign new parameters to the model
        model = {'U1': U1, 'c1': c1, 'U2': U2, 'c2': c2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model

# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda v: predict(model, v))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()

plt.figure(figsize=(8, 16))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
     plt.subplot(5, 2, i + 1)
     plt.title('Hidden Layer size %d' % nn_hdim)
     model = build_model(nn_hdim)
     plot_decision_boundary(lambda v: predict(model, v))
plt.show()