# Run some setup code for this notebook. Don't modify anything in this cell.

import random
import numpy as np
from cs224d.data_utils import *
import matplotlib.pyplot as plt

def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################
    
    ### YOUR CODE HERE
    import numpy as np
    
    # Get max for each row
    y = np.amax(x, axis=1)
    y = np.repeat(y, x.shape[1], axis=0)  
    y = np.reshape(y, x.shape)
    
    # rescale x to for numerical stability
    x1=np.exp(x-y)
    
    # normalize over the vector    
    x2 = np.sum(x1, axis=1)    # x2 is now the vector containing the row sums
    x2 = np.repeat(x2, x.shape[1], axis=0) 
    x2 = np.reshape(x2, x.shape)
        
    # normalize over the vector
    x2 = x1/x2
    
    ### END YOUR CODE
    
    return x2
    
def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################
    
    ### YOUR CODE HERE

    y = 1/(1+np.exp(-x))
    
    ### END YOUR CODE
    
    return y

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################
    
    ### YOUR CODE HERE
    
    f = f * (1-f)   
    
    ### END YOUR CODE
    
    return f
    
# Implement your skip-gram and CBOW models here
import random

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       # 
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    
    y = np.zeros(len(outputVectors))
    y[target] = 1
    
    z_3 = np.array([np.dot(outputVectors, predicted)])
    a_3 = softmax(z_3)
    cost = - np.sum(y * np.log(a_3))
    gradPred = np.dot((a_3 - y), outputVectors)
    grad = np.dot((a_3-y).T, np.array([predicted]))
        
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    
    # V = number of words in the corpus
    V = outputVectors.shape[0]
    #print "V", V
    #print "K", K
    
    # get the negative sampling tokens
    w_k = np.zeros(K)        
    #print "w_k", w_k
    
    # get target as one-hot vector
    target_vec = np.zeros(outputVectors.shape[0])
    target_vec[target] = 1
    
    # we are going to store the K one-hot negative sampling vectors in 
    # a matrix with dimensions K x V
    w_k_mat = np.zeros((K+1, V))
    w_k_mat[0,:] = target_vec
    #print "w_k_mat", w_k_mat
    
    for i in range(0, K):        
        tmp = dataset.sampleTokenIdx()        
        w_k[i] = tmp
        # make sure we sample somthing that is not the target
        while w_k[i] == target:
            w_k[i] = dataset.sampleTokenIdx()
            
        tmp = np.zeros(V)        
        tmp[w_k[i]] = 1
            
        # add it to the matrix       
        w_k_mat[i+1,:] = -tmp
       
       
    # outputVectors is of shape  V x n
    
    # select the relevant outputVectors (i.e. the w_k's
    # this should be a n x K matrix
    w_k_mat_Times_outputVectors = np.dot(w_k_mat, outputVectors)
    
    # now we multiply all the rows with \hat r = predicted to derive "- w_k^T \hat r"
    # this should be a vector with length K
    z3 =  np.dot(w_k_mat_Times_outputVectors, predicted)
    
    # now we apply the sigmoid function to all the elements
    a3 = sigmoid(z3)
    
    # and then the log function
    tmp4 = np.log(a3)
    
    # and we add that to the cost function
    cost = - np.sum(tmp4)
        
    delta_3 = a3 - 1 
    
    # delta_3 is (K x 1)
    # w_k_mat is (K x V)
    # outputVectors is (V x n)
    # to get z_3= w_k_mat x outputVectors x predicted
    # so now we need delta_3 x outputVectors is shape 
       
    gradPred = np.array([np.dot(delta_3,w_k_mat_Times_outputVectors)])
    #gradPred = np.array([np.zeros_like(predicted)    ])    
          
    tmp5 = np.dot(delta_3, w_k_mat)
    grad = np.dot(np.array([tmp5]).T, np.array([predicted]) )
    
    #print "gradPred", gradPred
    #print "grad", grad
    
    ### END YOUR CODE
    
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #         
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    
    debug = 0
    if debug:
        print "currentWord", currentWord
        print "C", C
        print "contextWords", contextWords
        print "tokens", len(tokens)
        print "inputVectors", inputVectors
        print "outputVectors", outputVectors
    
    V = len(tokens)
    
    # Generate 1 hot input vector
    x = np.zeros(V)
    x[tokens[currentWord]] = 1
    if debug:
        print "x", x
    
    # Get embedded word vectors
    u = np.dot(x, inputVectors)
    if debug:
        print "u", u
    
    # set h = u^(i)
    h = u

    # Generate 2C score vectors    
    #y = np.dot(outputVectors, h)
    #y = np.reshape(np.repeat(y, 2*C, axis=0), (V, 2*C)).T
    
    # y now is a matrix with 2C rows and V columns
    
    # Turn each score into probabilities  using softmax
    #y_hat = softmax(y)
    #if debug:
    #    print "y_hat", y_hat
    
    # those probabilities should match the "true" probabilities 
    # of the one-hot output vectors
    
    # cost = sum over all context words of softmaxCostAndGradient(predicted, target, outputVectors)
    
    cost = 0
    sumGradPred = np.zeros_like(h)
    sumGrad = np.zeros_like(outputVectors)
    expected_words =   np.array([])
    for cw in contextWords:
        #x = np.zeros(V)
        
        predicted = h # predicted = the middle vector
        target = tokens[cw]   # target is the index of the word we expect
                              
        cw_cost, cw_gradPred, cw_grad = word2vecCostAndGradient(h, target, outputVectors)
        if debug:
            print "cw_cost", cw_cost
            print "cw_gradPred", cw_gradPred
            print "cw_grad", cw_grad
        
        cost = cost + cw_cost
        sumGradPred = sumGradPred + cw_gradPred
        sumGrad = sumGrad + cw_grad    
    
    ### END YOUR CODE
    
    if debug:
        print "sumGradPred", sumGradPred
        print "sum(x)", np.sum(x)
    
    gradIn = np.dot(np.array([x]).T, sumGradPred)
    
    if debug:
        print "gradIn", gradIn
    
    gradOut = sumGrad
    
    if debug:
        print "gradOut", gradOut
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #         
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    
    debug = 0
    if debug:
        print "currentWord", currentWord
        print "C", C
        print "contextWords", contextWords
        print "tokens", tokens
        print "inputVectors", inputVectors
        print "outputVectors", outputVectors
    
    V = len(tokens)
    
    
    # iterate over all context words as input vectors
    h = np.zeros_like(inputVectors.shape[1])
    
    # lets use a matrix approach
    
    if 0: 
        for cw in contextWords:        
            x = np.zeros(V)
            x[tokens[cw]] = 1
            u = np.dot(x, inputVectors)
            h = h + u
    if 1:
        # we create a tmp matrix with shape (C x V)
        tmp = np.zeros((len(contextWords), V))
        i=0
        for cw in contextWords:        
            x = np.zeros(V)
            x[tokens[cw]] = 1
            tmp[i,:] = x
            i = i+1
            
        # then h should be tmp x inputVectors (shape C x n))
        # we than add it all up by multiplying 
        # h = 1(1, C) x (tmp x InputVectors) 
        
        one_vec = np.array([np.ones(len(contextWords))])
        
        t = np.dot(tmp, inputVectors)
        h = np.dot(one_vec, t)
        h = h.flatten()
    
    #print h
    #print h.flatten()
    
    # now we apply the softmaxCostAndGradient
        
    predicted = h # predicted = the middle vector
    target = tokens[currentWord]   # target is the index of the word we expect
                              
    cost, cw_gradPred, gradOut = word2vecCostAndGradient(h, target, outputVectors)
    
    # cw_Pread is shape (1,n)
    # one_vec is shape (1,C)
    # tmp is shape (C, V)
    # we need GradIn with shape (V x n)
    
    t = np.dot(one_vec.T,  cw_gradPred) # shape C x n
    t2 = np.dot(tmp.T, t) # shape V x n
    gradIn = t2
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut
    
# Gradient check!

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    
    debug = 0
    
    if debug: 
        print "ENTER word2vec_sgd_wrapper"
        print wordVectors
    
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        if debug:
            print "centerword", centerword
            print "context", context
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        if debug: 
            print "inputVectors", inputVectors.shape
            print "outputVectors", outputVectors.shape
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        
        if debug: 
            print "i", i
            print "c", c
            print "sum(gin)", np.sum(gin)
            print "sum(gout)", np.sum(gout)
        
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
    
    if debug: 
        print cost
        print grad
    #print i
        
    return cost, grad
    
    
# Now, implement SGD

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

import glob
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

import time

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    #   - PRINT_EVERY: specifies every how many iterations to output  #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################
    
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    t1 = time.time()
    
    for iter in xrange(start_iter + 1, iterations + 1):
        ### YOUR CODE HERE
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
        #print iter
        #print "x", x
        #print "f(x)", f(x)
        
        
        #print "cost", cost
        #print "sum(grad)", np.sum(grad)
        #import cProfile
        #cProfile.run('
        cost, grad = f(x)
               
        x = x - step * grad
        
        if iter % PRINT_EVERY == 0:
            t2 = time.time()
            print "i", iter, "cost", cost, "time", t2-t1, "est. remaining (min)", (iterations-iter) / PRINT_EVERY * (t2-t1) / 60
            t1 = time.time()
               
        #if iter == 20:
        #    break
        
        ### END YOUR CODE
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
            #break
            
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    
    return x
    
# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    #print "--- ENTER gradcheck_naive"
    #print f
    #import inspect
    #print inspect.getsource(f)
    
    print "- INITIAL CALL TO F"
    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    print "- ENTERING ITERATION"
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    cntr = 0
    while (cntr < 0):
        cntr += 1
        it.iternext() # Step to next dimension
    
    while not it.finished:
        ix = it.multi_index
    
        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        #print x[ix].shape
        #print x[ix]
        # print "working on index " 
        # print ix
        
        I = np.zeros_like(x)
        I[ix] = 1
        #print I.shape
        #print x
                   
        random.setstate(rndstate)
        yix = f(x-h*I)[0]
        
        random.setstate(rndstate)
        yix_plus_h = f(x+h*I)[0]
        
        #print "yix = %f, yix_plus_h = %f" % (yix, yix_plus_h)
        
        numgrad = (yix_plus_h - yix) / (2*h)
        #print "numgrad=%f" % numgrad
                
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            #print "Reldiff = %f" % reldiff
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"
    
def normalizeRows(x):
    """ Row normalization function """
    
    ### YOUR CODE HERE
    #print x
        
    # get current lengths of the rows in the matrix
    y = np.sqrt(np.sum(x * x, axis = 1))
    
    #print y
    
    # invert to get factor
    y = 1/y
    #print x.shape
        
    # repeat for pairwise multiplication, reshape into matrix and multiply by x
    x = np.reshape(np.repeat(y, x.shape[1]), x.shape) * x
   
    ### END YOUR CODE
    
    return x


if 0: 
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== For autograder ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
    exit()

#
# Load some data and initialize word vectors

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Train word vectors (this could take a while!)

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors, 
                              np.zeros((nWords, dimVectors))), axis=0)
#wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors, 
#                              (np.random.rand(nWords, dimVectors) - .5) / dimVectors), axis=0)

print "nWords", nWords
print wordVectors.shape

wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), 
                   wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# sanity check: cost at convergence should be around or below 10

# sum the input and output word vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print checkVecs