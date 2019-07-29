import numpy as np


def map_values(data, mapping):
    # maps values of numpy array using a mapping dictionary
    return np.vectorize(mapping.get)(data)


def sign(x):
    # implements np.sign function as sets random choice of 1 or -1 for 0 results
    x_out = np.sign(x)
    if x_out == 0:
        x_out = np.random.choice([-1, 1])
    return x_out


# load data into training and test numpy array, and dictionary into dict()
directory = 'data/'
files = [directory + file for file in ['pa3dictionary.txt', 'pa3test.txt', 'pa3train.txt']]
row_to_word = dict(enumerate(list(np.loadtxt(files[0], dtype='str'))))
train = np.loadtxt(files[1], dtype='int', delimiter=' ')
test = np.loadtxt(files[2], dtype='int', delimiter=' ')
num_features = train.shape[1] - 1

# map 2 / 1 labels to -1 / 1
mapping = {2: -1, 1: 1}
train[:, -1] = map_values(train[:, -1], mapping)
test[:, -1] = map_values(test[:, -1], mapping)


# simple perceptron class
class SimplePerceptron:
    def __init__(self):
        self.num_features = None
        self.w = None

    def fit(self, labeled_data, reset_model=True):
        # fit model to labeled_data, where
        #   labeled_data is 2d array, with columns as features and rows as samples
        #   final column is label, must be -1 or 1
        X = labeled_data[:, :-1]
        y = labeled_data[:, -1]

        if reset_model:
            self.num_features = X.size[1]
            self.w = np.zeros(self.num_features + 1)

        for i in range(y.size):
            x = np.append(X[i], [1])  # append 1 to the x, for bias term
            yhat = self.predict(x)  # estimate class using current weights
            if y[i] != yhat:  # if incorrect decision
                self.w = np.add(self.w, np.multiply(y[i], x))  # update weights

    def predict(self, x):
        return sign(np.dot(x, self.w))

    def test(self, labeled_data):
        # test classification of model using labeled_data, where
        #   labeled_data is 2d array, with columns as features and rows as samples
        #   final column is label, must be -1 or 1
        X = labeled_data[:, :-1]
        y = labeled_data[:, -1]

        num_errors = 0
        for i in range(y.size):
            x = np.append(X[i], [1])  # append 1 to the x, for bias term
            yhat = self.predict(x)  # estimate class
            if y[i] != yhat:  # if incorrect decision
                num_errors += 1
        return num_errors / y.size


# train model over 5 passes of the training data and report training & test error each time
simple_perceptron = SimplePerceptron()
print('## Simple Perceptron ##')
num_passes = 5
for i in range(num_passes):
    print('# Pass ' + str(i+1) + ' #')
    simple_perceptron.fit(train, reset_model=(i == 0))  # fit model and reset_model when i==0
    print('Training error: ' + '{:.4f}'.format(simple_perceptron.test(train)))
    print('Testing error:  ' + '{:.4f}'.format(simple_perceptron.test(test)))
    print('')
print('')
print('')


# kernelised perceptron class
class KernelisedPerceptron:
    def __init__(self):
        self.num_samples = None
        self.num_features = None
        self.alpha = None
        self.X_trained = None
        self.kernel = None

    def fit(self, labeled_data, reset_model=True, kernel=np.dot):
        # fit model to labeled_data using kernel, where
        #   labeled_data is 2d array, with columns as features and rows as samples
        #   final column is label, must be -1 or 1
        X = labeled_data[:, :-1]
        y = labeled_data[:, -1]

        if reset_model:
            self.kernel = kernel
            self.X_trained = X
            self.num_samples, self.num_features = X.shape
            self.alpha = np.zeros(self.num_samples)

        for j in range(self.num_samples):
            yhat = self.predict(X[j, :])
            if y[j] != yhat:
                self.alpha[j] += y[j]

    def predict(self, x):
        # classify sample x using trained model
        yhat = 0
        for i in range(self.num_samples):
            yhat += self.alpha[i] * self.kernel(self.X_trained[i, :], x)
        return sign(yhat)

    def test(self, labeled_data):
        # test classification using model on labeled_data, where
        #   labeled_data is 2d array, with columns as features and rows as samples
        #   final column is label, must be -1 or 1
        #   returns error rate
        num_errors = 0
        X = labeled_data[:, :-1]
        y = labeled_data[:, -1]

        for j in range(self.num_samples):
            yhat = self.predict(X[j, :])
            if y[j] != yhat:
                num_errors += 1

        return num_errors / self.num_samples


# kernel functions as defined in exercise
def kernel_exp(x1, x2):
    return np.exp(-np.linalg.norm(x1-x2) / 20)


def kernel_poly(x1, x2):
    return (np.dot(x1, x2) + 10) ** 2


kernel_perceptron = KernelisedPerceptron()

kernels = [np.dot, kernel_exp, kernel_poly]

messages = ['## Kernalised - dot product ##',
            '## Kernalised - exponential ##',
            '## Kernalised - polynomial ##']

# for each kernel, train model over 5 passes of the training data and report training & test error each time
num_passes = 5
for message, kernel in zip(messages, kernels):
    print(message)
    for i in range(num_passes):
        print('# Pass ' + str(i+1) + ' #')
        kernel_perceptron.fit(train, reset_model=(i == 0), kernel=kernel)  # fit model and reset_model when i==0
        print('Training error: ' + '{:.5f}'.format(kernel_perceptron.test(train)))
        print('Testing error:  ' + '{:.5f}'.format(kernel_perceptron.test(test)))
        print('')
    print('')
    print('')


