<center><h1>Validation and Evaluation</h1></center>

## Evaluation Techniques
- **k-Fold Cross Validation:** Split the dataset into k
partitions. Train the model independently k times, each time leaving a different partition out and use the left out partition as the test dataset.
[`sklearn.model_selection.KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
  - *Leave One Out:* Set k = n. Use all but one sample for training every time.
  - *Stratified k-Fold:* While generating the k folds, it tries to have approximately the same class ratios in folds to make each fold more representative.

## Sources of Error
- **Leakage**: Information about the labels leak into the training process in a way that wouldn't happen in the real world setting. This can include using features that are influenced by the outcome/label after the fact or including side-information that might be influenced by the label/annotation, which would not be available in real-world.

- **Sample Bias**: Discrepancy between the distributions that generate the train ing and real-world samples.
  - **Survivorship bias**: Focusing on a subset of subjects that are known to have success. For example, sampling companies that exist today and use their historical data for stock market predictions.

- **Overfitting:** The learning model is too complex for the available data and fits to the noise. Remedy: decrease model's complexity or (preferably) find more data

- **Underfitting:** The learning model is too simple for the available data and can't capture the underlying phenomenon we are trying to approximate.

- **Collinearity:** In multiple regression analysis, if one of the features is linearly associated/dependent on other features, there is collinearity between them. This phenomenon implies there is redundancy among the features and causes unreliability about the weight/impact of individual features on the predicted value.

- **Correlation:** Measures the linear dependence between two features. (Collinearity implies correlation but not vice versa.)

- **Heteroskedasticity:** The variance of error terms being different. Certain subsets of the feature space might be observed with different error dispersions than the others, confusing the models that treat them equally. 

## Performance Metrics
X: input space, Y: outcome space, A: action space, F: function space we are searching, l(.,.): loss function.

Let <math><mi>R</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>=</mo><mi>E</mi><mo>[</mo><mi>l</mi><mo>(</mo><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo><mi>y</mi><mo>)</mo><mo>]</mo></math> and <math><msub><mi>f</mi><mi>F</mi></msub><mo>=</mo><mi>a</mi><mi>r</mi><mi>g</mi><mi>m</mi><mi>i</mi><msub><mi>n</mi><mrow><mi>f</mi><mo>&#x2208;</mo><mi>F</mi></mrow></msub><mrow><mi>R</mi><mo>(</mo><mi>f</mi><mo>)</mo></mrow></math>.

- **Risk:** Expected loss of a new sample drawn from the same distribution (R(f)).
  - *Mean squared risk:* <math><mi>R</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>=</mo><mi>E</mi><mo>[</mo><mo>(</mo><mi>y</mi><mo>-</mo><mi>a</mi><msup><mi>)</mi><mn>2</mn></msup><mo>]</mo></math>
- **Bayes Risk:** Expected loss of Bayes decision function: <math><mi>a</mi><mi>r</mi><mi>g</mi><mi>m</mi><mi>i</mi><msub><mi>n</mi><mi>f</mi></msub><mo>{</mo><mi>R</mi><mo>(</mo><mi>f</mi><mo>)</mo><mo>}</mo></math>, <math><msup><mi>f</mi><mo>*</mo></msup><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>E</mi><mfenced open="[" close="]"><mrow><mi>y</mi><mo>|</mo><mi>x</mi></mrow></mfenced></math>

- **Empirical Risk:** Mean loss computed over the dataset (still assuming we are constrained to F). <math><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>n</mi></msub><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>l</mi><mo>(</mo><msub><mi>f</mi><mi>n</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo><mi>y</mi><mo>)</mo></math>.
  - *Empirical Risk Minimizer (ERM):* <math><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub><mo>=</mo><mi>a</mi><mi>r</mi><mi>g</mi><mi>m</mi><mi>i</mi><msub><mi>n</mi><mi>F</mi></msub><mo>{</mo><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>n</mi></msub><mo>)</mo><mo>}</mo></math>. It is random due to the randomness in dataset being used. 

- **Simple Performance Measures:**
  - *Accuracy:* (TP + TN) / (TP + TN + FP + FN)
  - *Precision:* TP/ (TP + FP)
  - *Recall/Sensitivity:* TP / (TP + FN)
  - *Specificity:* TN / (TN + FP)
  - *F1 score:* 2 * (precision * recall) / (precision + recall) = 2TP / (TP + FP + FN). 

- **AUC:** Area under the curve of ROC figure (FP vs. TP). It is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one (assuming 'positive' ranks higher than 'negative').

- **Coefficient of Determination:** <math><msup><mi>R</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>-</mo><mfrac><mrow><msub><mo>&#x2211;</mo><mi>i</mi></msub><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><mi>f</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>)</mo><msup><mi>)</mi><mn>2</mn></msup></mrow><mrow><msub><mo>&#x2211;</mo><mi>i</mi></msub><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><mover><mi>y</mi><mo>&#xAF;</mo></mover><msup><mi>)</mi><mn>2</mn></msup></mrow></mfrac></math>. It measures how well a model performs relative to the sample mean of the data. <math><msup><mi>R</mi><mn>2</mn></msup></math> = 1 indicates a perfect match, <math><msup><mi>R</mi><mn>2</mn></msup></math> = 0 indicates the model does no better than simply taking the mean of the data, and negative values mean even worse models.

## Excess Risk Decomposition
- **Approximation Error:** The loss due to constraining ourselves to F, (<math><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>F</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msup><mi>f</mi><mo>*</mo></msup><mo>)</mo></math>)
  - Shrinks with growing F.
  - Independent of the dataset, hence larger n doesn't have an impact.

- **Estimation Error:** The loss due to using empirical risk instead of true risk. (<math><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>F</mi></msub><mo>)</mo></math>)
  - Grows with growing F (for fixed n).
  - Shrinks with growing n. It is random, due to the randomness of the samples in the dataset.

- **Optimization Error:** The loss due to the suboptimality of our training/optimization process. We are essentially approximating <math><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub></math> due to factors like randomness in SGD, poor starting point, etc. It can be negative since we are only minimizing a random variable, but we might end up at a point that performs poorly at this instance but better in expectation. Let <math xmlns=<msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub></math> be our approximation of <math xmlns=<msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub></math>, which is our final model. (<math><mo>=</mo><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub><mo>)</mo></math>)

- **Excess Risk:** The total suboptimality. (Optimization Error + Estimation Error + Approximation Error) (<math><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msup><mi>f</mi><mo>*</mo></msup><mo>)</mo></math>)

## Bias-Variance Tradeoff
A *parameter* is a function of the distribution of a random variable, such as its mean or variance and hence is not random. A *statistic*, on the other hand is a function of a dataset, such as sample mean and sample variance, which is usually used as an estimate of an underlying parameter. The probability distribution of a statistic is called a *sampling distribution*.

As an example, consider the parameter <math><mi>&#x3BC;</mi><mo>:</mo><mi>P</mi><mo>&#x2192;</mo><mi mathvariant="normal">&#x211D;</mi></math> and the statistic <math><mover><mi>&#x3BC;</mi><mo>^</mo></mover><mo>:</mo><msub><mi>D</mi><mi>n</mi></msub><mo>&#x2192;</mo><mi mathvariant="normal">&#x211D;</mi></math>.

- **Standard error:** Standard deviation of sampling distribution.
- **Bias:** Bias<math><mo>(</mo><mover><mi>&#x3BC;</mi><mo>^</mo></mover><mo>)</mo><mo>=</mo><mi>E</mi><mfenced open="[" close="]"><mrow><mover><mi>&#x3BC;</mi><mo>^</mo></mover></mfenced><mo>-</mo><mi>&#x3BC;</mi></mrow></math>. An estimator is *unbiased* if it has zero bias. This is caused by an erroneous assumption in the learning model, causing underfitting.
- **Variance:** Var<math><mo>(</mo><mover><mi>&#x3BC;</mi><mo>^</mo></mover><mo>)</mo><mo>=</mo><mi>E</mi><mfenced open="[" close="]"><msup><mover><mi>&#x3BC;</mi><mo>^</mo></mover><mn>2</mn></msup></mfenced><mo>-</mo><mo>(</mo><mi>E</mi><mfenced open="[" close="]"><mover><mi>&#x3BC;</mi><mo>^</mo></mover></mfenced><msup><mi>)</mi><mn>2</mn></msup></math>. It measures the error from sensitivity to small fluctuations in the training set and hence high variance is an indicator of overfitting and unstability to the input samples.

Assuming an additive noise model (with <math><msup><mi>&#x3C3;</mi><mn>2</mn></msup></math> variance) where the observed samples are noisy observations from a function we want to approximate. Then, the squared loss can be decomposed as:
<math><mi>E</mi><mfenced open="[" close="]"><mrow><mo>(</mo><mi>y</mi><mo>-</mo><mover><mi>f</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><msup><mi>)</mi><mn>2</mn></msup></mrow></mfenced><mo>=</mo><mo>(</mo><mi>B</mi><mi>i</mi><mi>a</mi><mi>s</mi><mo>(</mo><mover><mi>f</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><msup><mi>)</mi><mn>2</mn></msup><mo>+</mo><mi>V</mi><mi>a</mi><mi>r</mi><mo>(</mo><mover><mi>f</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><mo>)</mo><mo>+</mo><msup><mi>&#x3C3;</mi><mn>2</mn></msup></math>.

One method to detect the bias-variance balance in the case of regression or soft classification is to check the model's <math><msup><mi>R</mi><mn>2</mn></msup></math> for training and test/validation sets.

- High variance models overfit and have very high training <math><msup><mi>R</mi><mn>2</mn></msup></math>, but low (even possibly negative) <math><msup><mi>R</mi><mn>2</mn></msup></math> on unseen data.
- High bias models underfit and perform similar <math><msup><mi>R</mi><mn>2</mn></msup></math> scores on both seen and unseen data since they do not fit to the noisy details of the data that would lead to a discrepancy.

Finding the sweet spot in bias-variance trade-off has inspired ensemble learning methods like random forest.

- **Decision Trees:** As the depth increases, variance increases and bias decreases.

- **Bagging (Bootstrapp Aggregation):** Bootstrap the dataset to generate, say N, independent datasets and train N independent tress. Then, use an aggragate method like majority vote for classification or average for regression to combine the outputs of these trees. Bagging works well with models that are low bias and high variance like trees.
  - *Out of Bag Samples:* The probability of a sample not show in a bootstrapped dataset converges to 1/e, so there is roughly a third of samples that do not show up on each particular bootstrapped dataset. One evaluation method is to aggregate the models that haven't seen a sample and test the model on this ensemble for testing.

- **Random Forests:** The goal is to do bagging with decision trees, but improve the performance by introducing randomness to reduce the correlation between trees. </br>
The idea is to use a random sampled subset of features (instead of all features) while making splitting decisions. This sampling is done independently for every node. (Typical values are in the order of square root of number of features) </br> 
Random forest incentivizes using deep and high variance trees (and hence overfit), but bagging helps with this issue and the ensemble ends up working well.

- **Boosting:** The general idea is to train the weak learners iteratively to focus more on more 'problematic' samples. The first learner is trained on the original data with equal weights, then the weights of sample errors are shifted to amplify the errors from samples that are believed to be poorly modeled by the previous learners. </br>
The general idea of this approach is to reduce the overall bias of the ensemble, which incentivizes using shallow and high bias trees.