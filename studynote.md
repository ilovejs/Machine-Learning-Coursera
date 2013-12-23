#   Machine Learning
<pre>
Notice:
    my school semester is nearing end, and projects and final exams are coming
    so for week 8 and 9, I'm quickly skimming course material (lecture slides without watching videos) 
    it turned out to be quite not bad, it require more thinking (filling in the gap myself, otherwise in the video, ideas are well explained)
    and I want to get a good score, so I still complete the review questions and programming assignments
    p.s. for week 8, I stopped before (140 - 5 - Choosing the Number of Principal Components (11 min).mp4)
</pre>
### XV. Anomaly Detection (Week 9)
### XVI. Recommender Systems (Week 9)
### Week8   XIII. Clustering (Week 8)
### Week8   XIV. Dimensionality Reduction (Week 8)
*   the meaning of eigenvectors, eigenvalue?
    *   programming exercise 2.2 Implementing PCA
### Week7   XII. Support Vector Machines
*   lecture
    *   logistic regression -> SVM
        *   kernel [linear kernel] [Gaussian kernel] [string kernel] [chi-square kernel]
            *   [Kernel Functions for Machine Learning Applications | ~/cesarsouza/blog](http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html)
        *   bias / variance analysis for SVM, C, sigma
            *   [Abductive Intelligence — Managing Bias - Variance Tradeoff in Machine Learning](http://blog.stephenpurpura.com/post/13052575854/managing-bias-variance-tradeoff-in-machine-learning)
    *   comparison and conditions of applyding 3 learning algorithms so far
*   programming exercise
    *   SVM
        *   Gaussian kernel
        *   choose best C and σ values.
    *   email spam, process text
        *   preprocessing
        *   vocabulary list, feature vector, top predictors for spam (weight)
    *   underlying
        *   function [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)
        *   function pred = svmPredict(model, X)
*   review question
    *   The SVM without any kernel (ie, the linear kernel) predicts output based only on θTx, so it gives a linear / straight-line decision boundary, just as logistic regression does.
    *   A neural network with many hidden units is a more complex (higher variance) model than logistic regression, so it is less likely to underfit the data.
    *   If the data are linearly separable, an SVM using a linear kernel will return the same parameters θ regardless of the chosen value of C (i.e., the resulting value of θ does not depend on C).
        *   A linearly separable dataset can usually be separated by many different lines. Varying the parameter C will cause the SVM's decision boundary to vary among these possibilities. For example, for a very large value of C, it might learn larger values of θ in order to increase the margin on certain examples.
### Week6   X. Advice for Applying Machine Learning 
*   video lecture
    *   X. Advice for Applying Machine Learning (Week 6)
    *   XI. Machine Learning System Design (Week 6)
*   programming exercise
    *   3.1 Learning Polynomial Regression
        *
        <pre>
            Keep in mind that even though we have polynomial terms in our feature
vector, we are still solving a linear regression optimization problem. The
polynomial terms have simply turned into features that we can use for linear
regression. We are using the same cost function and gradient that you wrote
for the earlier part of this exercise.
        </pre>
    *
    <pre>
        implementing and mastering the process
            Regularized Linear Regression CostFunction
            Regularized Linear Regression Gradient
            Learning Curve
                train error, cross validation on number of training examples
            Polynomial Feature Mapping
            Cross Validation Curve
                train error, cross validation on lambda

        the instruction contain good explaination of why, how and tips

        % An important concept in machine learning is the bias-variance tradeoff. Mod-
        % els with high bias are not complex enough for the data and tend to underfit,
        % while models with high variance overfit to the training data.

        % Recall that a learning curve plots training and cross validation error 
        % as a function of training set size.
    </pre>
*   review question
    *   Question 1
        *   high bias   <img src="/tmp/Chrome/10.1-b.png">
        *   high variance   <img src="/tmp/Chrome/10.1-c.png">
    *   Question 4
        *   Suppose you are training a regularized linear regression model. The recommended way to choose what value of regularization parameter λ to use is to choose the value of λ which gives the lowest test set error.    (You should not use the test set to choose the regularization parameter, as you will then have an artificially low value for test error and it will not give a good estimate of generalization error.)
        *   It is okay to use data from the test set to choose the regularization parameter λ, but not the model parameters (θ).    (   You should not use test set data in choosing the regularization parameter, as it means the test error will not be a good estimate of generalization error.)
        *   <strong >Suppose you are training a regularized linear regression model. The recommended way to choose what value of regularization parameter λ to use is to choose the value of λ which gives the lowest cross validation error.  (The cross validation lets us find the "just right" setting of the regularization parameter given the fixed model parameters learned from the training set.)</strong>
    *   Question 5
        *    If the training and test errors are about the same, adding more features will not help improve the results.    (If the two errors are the same, then the model has high bias, so adding more features will be helpful.)

### Week5   Neural Networks: Learning
* review
    *   Neural Networks <strong>intuition</strong>
* submit log
<pre>
    highlight
        debug process of backpropagation - debug.txt
        % ~~ debug ~~
        % embarrasing... 
            % the matrix order 25*10 or 10*25 (and now it's correct) D1 / D1'
            % because when it's unrolled, you can't tell
        % the bias entry... why not 1?
        % the m is divided into all terms
        % the origin of everthing
        % file:///media/EOS_DIGITAL/Machine%20Learning/screenshots/9%20-%202%20-%20Backpropagation%20Algorithm%20(12%20min).mp4-2013-11-25-09h02m54s157.png
    Feedforward
        decode y into original output of labels
        regularization can be added out of calculation (just add the term)
    backpropagation
        a bit tricky to think of the dimension of one matrix
            input, hidden, output layer dimension
            number of samples
        understand every step
            (map to original math equations one by one)
            calculation of delta (backpropagation)
                error reverely reflected on previous layer's weight
                gradient of each layer
            calculation of DELTA
                summation of the error with input
    tips & tricks
        checkNNGradients
            use a smaller matrix to test
</pre><img src="file:///media/EOS_DIGITAL/Machine%20Learning/screenshots/9%20-%202%20-%20Backpropagation%20Algorithm%20(12%20min).mp4-2013-11-25-09h02m54s157.png">
### Week4    Neural Networks: Representation
*   overview
    *   review before proceed
    *   learn from examples / applications
        *   neural network, multiple features
*   review
    *   interesting bio-mimic
        *   example of using other sensory and learn 
        <img src="screenshots/8 - 2 - Neurons and the Brain (8 min).mp4-2013-11-19-08h18m40s147.png">
    *   neural network
        *   activation, weights
        <img src="screenshots/8 - 3 - Model Representation I (12 min).mp4-2013-11-19-08h32m30s0.png">
        <img src="screenshots/8 - 4 - Model Representation II (12 min).mp4-2013-11-19-08h37m37s247.png">
*   review question
    *   multi-class classification output (The outputs of a neural network are not probabilities, so their sum need not be 1.)
*   programming exercise
    *   neural network
        *   activation don't forget sigmoid function
### Week3 Logistic Regression & Regularization
####    Logistic Regression
- ->  problem of Linear Regression for classification
- ->  Logistic Regression (change expression of hypothesis)
- ->  Sigmoid Function (Logistic function) & Decision Boundary
- ->  Cost Function (non-convex -> convex)
- ->  simplify Cost Function, Gradient Descent
- ->  * Advanced Optimization
    *   fminunc
    *   Optimization Algorithm: Gradient descent, Conjugate gradient, BFGS  , L-BFGS
- ->  Multiclass Classification: one vs all
*   Question
    *   1/2m in Cost Function of Linear Regression, where does 1/2 come from?
    *   Logistic Regression Gradient 求导 (形式与Linear Regression相似)
####    Regularization
->  overfitting / underfitting
->  regularized Linear / Logistic Regression
-   normal equation

*   Question
    *   lamda 如何选取, weighted?
    *   lamda 在 normal equation 中
####    Programming Exercise
*   plotting
    *   find index, plot, mark legend in the same order as plot
### Week2   Linear Regression with Multiple Variables
####    Programming Exercise

  

