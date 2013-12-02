#   Machine Learning
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

  

