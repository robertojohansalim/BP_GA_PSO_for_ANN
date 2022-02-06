# %%
# https://www.knowledgehut.com/blog/data-science/types-of-classification-in-ml
from sklearn.datasets import make_blobs 
from collections import Counter 
from numpy import where 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

def load_datasets(datasets_name="wine"):
    if datasets_name == "iris":
        dataset = datasets.load_iris()
    elif datasets_name == "wine":
        dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target

    y_ = y.reshape(-1, 1) # Convert data to a single column

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)

    # Split the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return x_train, x_test, y_train, y_test

def saveReport(accuracy_history):
    data = {
        "accuracy_history": accuracy_history
    }
    pickle.dump(data, open("./GA_report.pkl", "wb"))

x_train, x_test, y_train, y_test = load_datasets()


# %%

import random
import numpy as np
import pandas as pd
import copy
import time
from numpy import exp
from sklearn.preprocessing import OneHotEncoder

random.seed(42)
np.random.seed(42)

class Network(object):

    def __init__(self, sizes):
        
        '''The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.'''

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])

    def feedforward(self, a):
        '''Return the output of the network if ``a`` is input.'''
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
            # a = np.dot(w,a)+b
        # a = self.softmax(a)
        return a

    # Actifation Functions
    def softmax(self, z):
        e = exp(z)
        e_sum = e.sum()
        if e_sum == 0:
            e_sum = 1
        return e / e_sum

    def sigmoid(self, z):
        '''The sigmoid function.'''
        return 1.0/(1.0+np.exp(-z))

    def relu(self, z):
        def relu_func(x):
            return x if x > 0 else 0
        return np.array(list(map(relu_func, z)))

    def score(self, X, y):

        '''
        @X = data to test
        @y = data-label to test
        @returns = score of network prediction (less is better)
        @ref: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
        '''

        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            score = np.sum(np.power(predicted-actual,2))
            # print("Accuracy Argmax:", np.argmax(predicted), np.argmax(actual), "Score:",score)
            # txt ="[ "
            # for pred in predicted:
            #     txt += "{}, ".format(pred)
            # print(txt, "]", sum(predicted))
            total_score += score  # mean-squared error
        # total_score /= X.shape[0] 
        # print("Total Score:", total_score)
        return total_score

    def accuracy(self, X, y):

        '''
        @X = data to test
        @y = data-label to test
        @returns = accuracy (%) (more is better)
        '''

        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1,1))
            accuracy += int(np.argmax(output) == np.argmax(y[i]))
        return (accuracy / X.shape[0]) * 100

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s

class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):

        '''
        n_pops   = How much population do our GA need to create
        net_size = Size of neural network for population members
        mutation_rate = probability of mutating all bias & weight inside our network
        crossover_rate = probability of cross-overing all bias & weight inside out network
        retain_rate = How many to retain our population for the best ones
        X = our data to test accuracy
        y = our data-label to test accuracy
        '''

        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]
    
    def get_random_point(self, type):

        '''
        @type = either 'weight' or 'bias'
        @returns tuple (layer_index, point_index)
            note: if type is set to 'weight', point_index will return (row_index, col_index)
        '''

        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers-2), 0
        if type == 'weight':
            row = random.randint(0,nn.weights[layer_index].shape[0]-1)
            col = random.randint(0,nn.weights[layer_index].shape[1]-1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0,nn.biases[layer_index].size-1)
        return (layer_index, point_index)

    def get_all_scores(self):
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_accuracy(self):
        return [net.accuracy(self.X, self.y) for net in self.nets]

    def crossover(self, father, mother):

        '''
        @father = neural-net object representing father
        @mother = neural-net object representing mother
        @returns = new child based on father/mother genetic information
        '''

        # make a copy of father 'genetic' weights & biases information
        nn = copy.deepcopy(father)

        # cross-over bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # replace genetic (bias) with mother's value
            if random.uniform(0,1) < self.crossover_rate:
                nn.biases[layer][point] = mother.biases[layer][point]

        # cross-over weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # replace genetic (weight) with mother's value
            if random.uniform(0,1) < self.crossover_rate:
                nn.weights[layer][point] = mother.weights[layer][point]
        
        return nn
        
    def mutation(self, child):

        '''
        @child_index = neural-net object to mutate its internal weights & biases value
        @returns = new mutated neural-net
        '''

        nn = copy.deepcopy(child)

        # mutate bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # add some random value between -0.5 and 0.5
            if random.uniform(0,1) < self.mutation_rate:
                nn.biases[layer][point] += random.uniform(-0.5, 0.5)

        # mutate weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # add some random value between -0.5 and 0.5
            if random.uniform(0,1) < self.mutation_rate:
                nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)

        return nn

    def evolve(self):

        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])
        # print("Sorted Score List")
        # print("Sorted Score:", [obj[1] for obj in score_list])
        # print()
        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]

        # keep only the best one
        retain_num = int(self.n_pops*self.retain_rate)
        score_list_top = score_list[:retain_num]

        # return some non-best ones
        retain_non_best = int((self.n_pops-retain_num) * self.retain_rate)
        for _ in range(random.randint(0, retain_non_best)):
            score_list_top.append(random.choice(score_list[retain_num:]))

        # breed new childs if current population number less than what we want
        while len(score_list_top) < self.n_pops:

            father = random.choice(score_list_top[:retain_num + retain_non_best])
            mother = random.choice(score_list_top[:retain_num + retain_non_best])

            if father != mother:
                new_child = self.crossover(father, mother)
                new_child = self.mutation(new_child)
                score_list_top.append(new_child)
        
        # copy our new population to current object
        self.nets = score_list_top

# %%
X = x_train
y = y_train

print(X.shape)
print(y.shape)

# parameters
N_POPS = 30
NET_SIZE = [X[0].shape[0],20,y[0].shape[0]] 
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.4
RETAIN_RATE = 0.4

EPOCH = 1000

# start our neural-net & optimize it using genetic algorithm
nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, X, y)

start_time = time.time()

# run for n iterations
accuracy_history = []
# for i in range(1000):
global_prev_best_accuracy = 0
max_count_after_best = 100
count_after_best = 0
for i in range(EPOCH):

    # accuracy = 
    best_current_accuracy = max(nnga.get_all_accuracy())
    if i % 1 == 0:
        print("Current iteration : {}".format(i+1))
        print("Time taken by far : %.1f seconds" % (time.time() - start_time))
        print("Current top member's network accuracy: %.2f%%\n" %best_current_accuracy)
        # print(nnga.get_all_accuracy())
        
    accuracy_history.append(best_current_accuracy)
    # if global_prev_best_accuracy < best_current_accuracy:
    #     global_prev_best_accuracy = best_current_accuracy
    #     count_after_best = 0
    # else:
    #     count_after_best += 1
    
    # if count_after_best > max_count_after_best:
    #     print("Early Stopped after", count_after_best, "without improvemnet from: ", global_prev_best_accuracy, "Current was:", best_current_accuracy)
    #     break

    # evolve the population
    nnga.evolve()

import numpy as np
import matplotlib.pyplot as plt
plt.plot(accuracy_history)
plt.show()


saveReport(accuracy_history)


