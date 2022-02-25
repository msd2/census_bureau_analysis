# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A random forest classifier trained to predict whether a person's salary is above or below $50k based on various features about that person.

## Intended Use
The model is a practice project without an intended in-production use.

## Training Data
Dataset contains 14 independent variables and 1 dependent variable.
6 variables are numeric, the rest are categorical.
There are 32,561 entries.

## Evaluation Data
20% of the data is used for evaluation.

## Metrics
Metrics used are fbeta, precision and recall.

fbeta: 0.723
precision: 0.641
recall: 0.680

## Ethical Considerations
The model should not be used to decide someone's salary. This is because there are many features not captured in our dataset which must be considered.

## Caveats and Recommendations
The model could be improved upon significantly by doing a grid search to optimise the hyperparameters.