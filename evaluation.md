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

- **Approximation Error:** The loss due to constraining ourselves to F, (<math><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>F</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msup><mi>f</mi><mo>*</mo></msup><mo>)</mo></math>)
  - Shrinks with growing F.
  - Independent of the dataset, hence larger n doesn't have an impact.

- **Estimation Error:** The loss due to using empirical risk instead of true risk. (<math><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msub><mi>f</mi><mi>F</mi></msub><mo>)</mo></math>)
  - Grows with growing F (for fixed n).
  - Shrinks with growing n. It is random, due to the randomness of the samples in the dataset.

- **Optimization Error:** The loss due to the suboptimality of our training/optimization process. We are essentially approximating <math><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub></math> due to factors like randomness in SGD, poor starting point, etc. It can be negative since we are only minimizing a random variable, but we might end up at a point that performs poorly at this instance but better in expectation. Let <math xmlns=<msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub></math> be our approximation of <math xmlns=<msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub></math>, which is our final model. (<math><mo>=</mo><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>^</mo></mover><mi>n</mi></msub><mo>)</mo></math>)

- **Excess Risk:** The total suboptimality. (Optimization Error + Estimation Error + Approximation Error) (<math><mi>R</mi><mo>(</mo><msub><mover><mi>f</mi><mo>~</mo></mover><mi>n</mi></msub><mo>)</mo><mo>-</mo><mi>R</mi><mo>(</mo><msup><mi>f</mi><mo>*</mo></msup><mo>)</mo></math>)