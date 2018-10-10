# poverty_prediction
Playing around with multiclass classification of Costa Rican household dataset (https://www.kaggle.com/c/costa-rican-household-poverty-prediction)

## For model development, see
`/notebooks/10_01_model_development.ipynb`
- Algorithm comparison is /notebooks/2018_09_12_compare_algorithms.ipynb

## Important note: 
My goal with this repository is **not to get the best Kaggle score**. I know, crazy, right? I'm more interested in learning best practices, such as building one pipeline for the entire model. Most if not all of the leading kernels (such as https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough) may do pipelines for a couple steps but not for the whole model. Indeed, I scored higher when I did all the transformers (including feature selection) once then scored cross validation on just the final estimator (rather than the entire pipeline).  

## Outline + Lessons (see `/notebooks/10_01_model_development.ipynb`)
- **Feature engineering:** 
    - Manually using logical features hurt the model, the best way seemed to be making many features and automating feature selection (or using a properly regularized estimator) 
- **Learning algorithm comparison**
- **Feature selection:** 
    - `RFECV` seems to be super sensitive to settings, including the classifier used and hyperparameters therein. Even within cross-validation folds (ie, using the same classifier and hyperparameters), I saw between 13 and 143 features selected. This merits further investigation before using in a final pipeline.
    - `SelectFromModel` is much quicker and consistent than `RFECV`, especially given that it exposes an important hyperparameter (`threshold`) for hyperparameter tuning. User note: make sure your feature importances are scaled appropriately for your `threshold`. Oddly, using the final estimator (`BaggedLGBMClassifier`) as the `SelectFromModel` classifier resulted in worse cross-validation scores. 
- **Early stopping + preliminary model evaluation**
    - This was key, though it still results in overfitting
- **Bagging**
    - For simple (and untuned) classification algorithms, bagging had inconsistent results on bias and variance. However, for my implementation of `BaggedLGBMClassifier`, it significantly reduced both bias and variance. Hooray!
    - I ended up writing a **custom sklearn estimator**, `BaggedLGBMClassifier`, because I couldn't figure out a way to use bagging *and* early_stopping with the sklearn API (e.g. `BaggingClassifier`). My implementation uses bagging of 5 `LGBMClassifier` estimators whose early_stopping is determined using the unsampled (aka "out-of-bag") observations as validation set (since bootstrapped sampling of the data leaves ~37% of the data unsampled).
- **Model performance in pipeline**
- **Hyperparameter tuning** 
    - I had mixed success using `Hyperopt` - while it takes a little more setup than `GridSearchCV`, I found it to be very useful and much quicker than `GridSearchCV`. 
    - *However,* I'm curious whether using the base implementation of `Hyperopt` can be dangerous. Unlike `GridSearchCV`, it uses only one train/test split of the data to find optimal hyperparameters - couldn't this lead to overfitting? I'm hoping that my use of bagging for the final estimator will avoid or mitigate potential overfitting. 
    
    
## Next steps
- I haven't yet used sophisticated insights tools for my `BaggedLGBMClassifier`, such as permutation importance, partial dependence plots, and SHAP values. Here is a notebook I've used these tools for a simple Random Forest classifier: https://github.com/zwrankin/chicago_bicycle_share/blob/master/notebooks/2018_09_24_initial_data_exploration_and_models.ipynb
- One of the obstacles to implementing these tools is figuring out how to get feature names out of pipelines with feature selection. 
- Build ensembles with `brew` (https://pypi.org/project/brew/) or `mlens` (https://github.com/flennerhag/mlens)
 
 
 ## Running model: 
 Here is how to fit the tuned model outside of the notebook
```
import os
import sys
import pandas as pd
import pickle

# Add library to path 
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from lib.model import pipeline, pipeline_cv_score

X_data = pd.read_hdf('../models/data.hdf', key='X_data')
y_data = pd.read_hdf('../models/data.hdf', key='y_data')
tuned_params = pickle.load(open("../models/tuned_params.p", "rb"))

pipeline.set_params(**tuned_params)

pipeline_cv_score(pipeline, X_data, y_data)

# pipeline.fit(X_data)
# pipeline.predict(TEST_FEATURES_HERE)
```

