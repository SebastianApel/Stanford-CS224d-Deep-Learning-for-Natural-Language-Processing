% Problem Set 1 - Solutions

<script> setTimeout(function(){   window.location.reload(1);}, 5000); </script>



## 1. Softmax

Prove that softmax is invariant to constant offsets in the input, i.e. $softmax(x) = softmax(x+c)$

**Proof**

Let $x= (x_1, ..., x_n)$. Using the softmax definition:

$$softmax(x+c) = \begin{bmatrix} \vdots \\ \frac{exp(x_j+c)}{\sum_{i=1}^{n}exp(x_i+c)} \\ \vdots  \end{bmatrix}$$

Now we look at the element j in detail:

$$\begin{align*}
softmax_j(x+c)
&=\frac{exp(x_j+c)}{\sum_{i=1}^{n}exp(x_i+c)} \\
&= \frac{exp(c)exp(x_j)}{\sum_{i=1}^{n}exp(x_i)exp(c)} \\
&= \frac{exp(c)exp(x_j)}{exp(c)\sum_{i=1}^{n}exp(x_i)} \\
&= \frac{exp(x_j)}{\sum_{i=1}^{n}exp(x_i)} \\
&= softmax_j(x) \\
\end{align*}
$$

Since this holds true for all $j$, we have our proof.

## 2. Neural Netword Basics

### 2a. Sigmoid Gradients 

Derive the gradients of the sigmoid function and show that it can be rewritten as a function of the
function value.

**Proof**

Let's start with the definition of $\sigma(x) = \frac{1}{1+e^{-x}}$ and we substitute $\sigma(x) = \frac{1}{z}$ with $z=1+e^{-x}$.

Now, we look at the derivative:

$$\begin{align*}
\frac{d\sigma(x)}{dx} 
&= \frac{d\sigma(x)}{dz}  \frac{dz}{dx} \\
&= \frac{d}{dz} \frac{1}{z} \frac{d}{dx} 1+e^{-x}\\
&= - \frac{1}{z^2} (- e^{-x})\\
&= \frac{1}{{1+e^{-x}}^2}  e^{-x}\\
&= \frac{1}{{1+e^{-x}}}\frac{{e^{-x}}}{{1+e^{-x}}}\\
&= \frac{1}{{1+e^{-x}}}(\frac{{e^{-x}}}{{1+e^{-x}}})\\
&= \frac{1}{{1+e^{-x}}}(\frac{{1+e^{-x}-1}}{{1+e^{-x}}})\\
&= \frac{1}{{1+e^{-x}}}(\frac{{1+e^{-x}}}{{1+e^{-x}}}-\frac{1}{{1+e^{-x}}})\\
&= \frac{1}{{1+e^{-x}}}(1-\frac{1}{{1+e^{-x}}}) \\
&= \sigma(x) (1-\sigma(x))\\
\end{align*}
$$

There we go.



### 2b. Softmax-CE gradient
Derive the gradient with regard to the inputs of a softmax function when cross entropy loss is used for
evaluation.

The derivative of the Cross Entropy Error with respect to $w_{a,b}$:
    



$$softmax(\theta) = \begin{bmatrix} \vdots \\ \frac{exp(\theta_j)}{\sum_{i=1}^{n}exp(\theta_i)} \\ \vdots  \end{bmatrix} 
\text{. and } CE(y,\hat y)=-\sum y_i log(\hat y_i) \text{ with } \hat y = softmax(\theta)$$

We want to get $\frac{dCE}{d\theta}$. Lets look at the individual elements of the $\theta$. 
And let's assume $o$ is the index of the "hot" element in the one-hot vector $y$.

$$\begin{align*}
\frac{dCE}{d\theta_k} 
&= \frac{d}{d \theta_k} - \sum y_i log( softmax_i(\theta) )  \\
&= \frac{d}{d \theta_k} - \sum y_i log( softmax_i(\theta) )  \\
&= \frac{d}{d \theta_k} - \sum y_i  log( \frac{exp(\theta_i)}{\sum_{j=1}^{n}exp(\theta_j)}  )  \\
&= \frac{d}{d \theta_k} - \sum (y_i \theta_i + log({\sum_{j=1}^{n}exp(\theta_j)}  ) ) \\
&= \frac{d}{d \theta_k} + \sum (y_i \theta_i ) - \sum (log({\sum_{j=1}^{n}exp(\theta_j)}  ) ) \\
&= \frac{d}{d \theta_k} + \sum (y_i \theta_i ) - \frac{d}{d \theta_k} \sum (log({\sum_{j=1}^{n}exp(\theta_j)}  ) ) \\
&= - y_k  + \frac{d}{d \theta_k} \sum (log({\sum_{j=1}^{n}exp(\theta_j)}  ))  \\
&= - y_k + \sum \frac{d}{d \theta_k}  log({\sum_{j=1}^{n}exp(\theta_j)} )    \\
&= - y_k   + \sum  \frac{d}{d \theta_k} log({\sum_{j=1}^{n}exp(\theta_j)} )    \\
&= - y_k   + \sum  \frac{1}{\sum_{j=1}^{n}exp(\theta_j)} \frac{d}{d \theta_k} {\sum_{j=1}^{n}exp(\theta_j)}     \\
&= - y_k   + \sum  \frac{1}{\sum_{j=1}^{n}exp(\theta_j)} \frac{d}{d \theta_k} {exp(\theta_k)}     \\
&= - y_k   +   \frac{1}{\sum_{j=1}^{n}exp(\theta_j)} \sum \frac{d}{d \theta_k} {exp(\theta_k)}     \\
&= - y_k   +  \frac{1}{\sum_{j=1}^{n}exp(\theta_j)} exp(\theta_k)     \\
&= - y_k   + softmax_k(\theta)   \\
&=  softmax_k(\theta) - y_k   \\
\end{align*}$$

So the  $\frac{dCE}{d\theta}=softmax(\theta)-y$.



### 2c. Gradients with respect to x

Derive the gradients with respect to the inputs x

We take the nameing definitions from the Lecture Notes Nr. 3, i.e.
$$\begin{align*}
& x = z^{(1)} = a^{(1)}  & \text{with dimensions} 1 \times n \\
& z^{(2)} = xW^{(1)}+b^{(1)} & \text{with dimensions of W of } n \times m \text{ and b of} 1 \times m \\
& a^{(2)} = \sigma(z^{(2)}) & \text{with dimensions of } 1 \times m \\
& z^{(3)} = xW^{(2)}+b^{(2)} & \text{with dimensions of W of } m \times o \text{ and b of} 1 \times o \\
& a^{(3)} =  softmax(z^{(2)}) & \text{with dimensions of } 1 \times o \\
& CE(y,a^{(3)}) = -\sum y_i a_i^{(3)}
\end{align*}$$

In the previous excersise, we have derived the error signal at $z^{(3)}$ to be $\delta^{(3)} = a^{(3)} - y$.

1. At $z^{(3)}$, the error signal is $\delta^{(3)} = a^{(3)} - y$ and has dimension $(1 \times o)$
2. At $a^{(2)}$, the error signal is $W^{(2)}^T \delta^{(3)} $ and has dimensions $(1 \times m)$
3. At $z^{(2)}$, the error signal is $\delta^{(2)} = (W^{(2)}^T \delta^{(3)})\circ \sigma'(z^{(2)}) $ and has dimensions $(1 \times m)$
4. At $x=a^{(1)}$, the error signal is $W^{(1)}^T \delta^{(2)} $ and has dimensions $(1 \times n)$

So, the gradients with respect to x is given in the line above.

### 2d. Number of Parameters

How many parameters are there in this neural network, assuming the input is $D_x$-dimensional, the
output is $D_y$-dimensional, and there are H hidden units?

Answer: There are $k = (D_x H) + H + (H D_y) + D_y$ number of parameters. 

Proof: See the dimensions of $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}$ in Problem 2.c.
</pre>

## 3. Word2Vec

### 3a. Gradients with respect to $\hat r$

Definitions:
Let $C$ be the number of context words in both directions.
Let $V$ be the number of words in the vocabulary.
Let $n$ be the length of vector in the hidden layer.

Let $\hat r$ be a vector of length $n$ (is equal to $h$ in the lecture notes).

Let $outputVectors$ be a matrix of dimension $n \times V$

Let $cost$ be the Softmax-CE cost.

Let $y$ be the expected result, a one-hot vector with length V.

Let $t$ be the "target" index, at which y is "hot"

Let $\theta$ be the result of the $outputVectors^T \hat r$

$$
\begin{align*}
cost &= CE(y, softmax(\theta))  \\
&= - \sum_{i=1}^V y_i log (softmax_i(\theta)) \\
&= - y_t log (softmax_t(\theta)) & \text{since y is a one hot-vector}\\
&= - log (softmax_t(\theta)) & \text{since y_t is a one }\\
&= - log (\frac{exp(\theta_t)}{\sum_{i=1}^{V}exp(\theta_i)}) & \text{applied softmax definition}\\
&= - (log (exp(\theta_t)) - log(\sum_{i=1}^{V}exp(\theta_i)) & \text{log of fraction }\\
&= - \theta_t + log(\sum_{i=1}^{V}exp(\theta_i)) & \text{log(exp())}\\
\end{align*}
$$

If we apply the structure from before, then

At $z^{(3)}$, $\delta^{(3)} = \frac{d CE}{d\theta} = softmax(\theta) - y$.

At $a^{(2)} = \delta^{(3)} outputVectors^T$.

Since the transfer function is the identity function, $z^{(2)} = a^{(2)}$.

The gradient with respect to $\frac{d CE}{\hat r}=softmax(\hat r outputVectors) - y$

## 3b. The gradient with respect to $outputVectors$

The gradient with respect to $outputVectors$ is $a^{(2)} delta^{(3)}$, i.e. $\frac{d CE}{d outputVectors}=\hat r (softmax(\hat r outputVectors) - y)$

## 3c. Gradients when using negative sampling

When using negative sampling, we assume that we draw $K$ negative samples and use a cost function $J$ as defined like following:
$J(\hat r, w_i, w_{1..K} ) = - log(\sigma(w_i^T, \hat r))   - \sum_{k=1}^{K} log(\sigma(-w_k^T\hat r))$.

We want to derive the derivatives of $J$ with regard to $\hat r$ and to $outputVectors$.

As a reminder:  $\sigma(x) = \frac{1}{1+exp(-x)}$

Lets define $\theta$ as a vector of length $K+1$ with
$\theta = \begin{pmatrix} w_i^T\hat r \\ - w_1^T\hat r \\ \vdots \\ -w_K^T\hat r \end{pmatrix} $

Then we can rewrite $J$ as $J = - \sum_{k=1}^{K+1} log(\sigma(\theta_k))$

We derive 

$$
\begin{align*}
\frac{d J}{d \theta_i} &= \frac{d}{d\theta_i} - \sum_{k=1}^{K+1} log (\sigma(\theta_k)) \\
&= - \sum_{k=1}^{K+1} \frac{d}{d\theta_i}  log (\sigma(\theta_k)) \\
&= - \sum_{k=1}^{K+1} \frac{d}{d\sigma}  log (\sigma(\theta_k)) \frac{d}{d\theta_i} \sigma(\theta_k) & \text{Chain Rule}\\
&= - \sum_{k=1}^{K+1} \frac{1}{\sigma(\theta_k)} \frac{d}{d\theta_i} 
\sigma(\theta_k) & \text{derivative of log}\\
&= - \sum_{k=1}^{K+1} \frac{1}{\sigma(\theta_k)} \frac{d}{d\theta_i} 
\sigma(\theta_k) & \text{for } i \ne k \text{, } \frac{d}{\theta_i}\sigma(\theta_k) = 0\\
&= - \frac{1}{\sigma(\theta_i)} \frac{d}{d\theta_i} 
\sigma(\theta_i) & \text{for }\\
&= - \frac{1}{\sigma(\theta_i)} \sigma(\theta_i) (1-\sigma(\theta_i)) & \text{used the derivative of } \sigma(x)\\
&= -  (1-\sigma(\theta_i)) & \\
&= \sigma(\theta_i) -1 & \\
\end{align*}
$$

and 

$$
\begin{align*}
\frac{d J}{d \theta_i} \frac{d \theta_i}{d\hat r_p} 
&= ((\sigma(\theta_i) -1) \frac{d \theta_i}{d\hat r_p} \\
&= ((\sigma(\theta_i) -1) \frac{d}{d\hat r_p}  w_i^T\hat r \\
&= ((\sigma(\theta_i) -1) \frac{d}{d\hat r_p}  \sum_{j=1}^{K+1}((w_i^T)_j\hat r_j) \\
&= (\sigma(\theta_i) -1) (w_i^T)_p \\
\end{align*}
$$


So $z^{(3)} 
= \begin{pmatrix} w_i^T\hat r \\ - w_1^T\hat r \\ \vdots \\ -w_K^T\hat r \end{pmatrix}
= \begin{pmatrix} w_i^T \\ - w_1^T \\ \vdots \\ -w_K^T r \end{pmatrix} \hat r
$. 

NOT FULLY SOLVED HERE - see code.

This cost function is O(k) and therefore independant for the vocabulary size, whereas the softmax-CE cost function is O(V), and therefore dependant on the vocabulary size.

## 3d. 

See implementation



## 4a. Why Regularization 

Regularization in the cost function incentivises that all weights are small.
Because of this, no single parameter dominates the prediction, and the predictor is less likely to overfit.

## 4b. Plot regularization against classification accuracy

not done
