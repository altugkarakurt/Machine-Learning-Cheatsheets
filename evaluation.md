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

## Performance Metrics
X: input space, Y: outcome space, A: action space, F: function space we are searching, l(.,.): loss function.

Let R(f) = E[l(f(x), y)] and f\_F = (argmin_{f in F} {R(f)})

- **Risk:** Expected loss of a new sample drawn from the same distribution (R(f)).
  - *Mean squared risk:* R(f) = E[(y-a)^2]
- **Bayes Risk:** Expected loss of Bayes decision function (argmin_f {R(f)}), f*(x) = E[y|x]

- **Empirical Risk:** Mean loss computed over the dataset (still assuming we are constrained to F). (R(f_n) = 1/n sum\_{i = 1}^n l(f\_n(x), y))
  - *Empirical Risk Minimizer (ERM):* \hat{f}\_n = argmin\_F {R(f_n)}. It is random due to the randomness in dataset being used. 

- **Approximation Error:** The loss due to constraining ourselves to F (R(f\_F) - R(f\*))
  - Shrinks with growing F.
  - Independent of the dataset, hence larger n doesn't have an impact.

- **Estimation Error:** The loss due to using empirical risk instead of true risk. (R(\hat{f}\_n) - R(f\_F))
  - Grows with growing F (for rixed n).
  - Shrinks with growing n. It is random, due to the randomness of the samples in the dataset.

- **Optimization Error:** The loss due to the suboptimality of our training/optimization process. We are essentially approximating \hat{f}\_n due to factors like randomness in SGD, poor starting point, etc. It can be negative since we are only minimizing a random variable (\sum...), but we might end up at a point that performs poorly at this instance but better in expectation (E[\sum...s]). Let \tilde{f}\_n be our approximation of \hat{f}\_n, which is our final model. (= R(\tilde{f}\_n) - R(\hat{f}\_n))

- **Excess Risk:** The total suboptimality. (Optimization Error + Estimation Error + Approximation Error) (R(\tilde{f}\_n) - R(f\*))