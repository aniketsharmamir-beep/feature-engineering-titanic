# Feature Engineering and Selection Report

## Dataset
Cleaned Titanic dataset (891 passengers) from an earlier cleaning task. The goal is to predict whether a passenger survived.

## Feature Engineering
New features created from existing columns:
- FamilySize: siblings/spouses + parents/children + 1
- IsAlone: 1 if the passenger travelled alone
- FarePerPerson: Fare divided by FamilySize
- Title: extracted from Name (Mr, Mrs, Miss, Master, Rare)
- AgeGroup: Age grouped into Child, Teen, Adult, Senior

Categorical variables (Sex, Embarked, Title, AgeGroup) were one-hot encoded, and identifier columns (PassengerId, Name, Ticket) were dropped. This gave 18 features.

## Scaling
The numerical columns (Age, Fare, SibSp, Parch, FamilySize, FarePerPerson) were standardized to mean 0 and standard deviation 1. The scaler was fitted on the training set only, to avoid leaking information from the test set.

## Feature Selection
Method: SelectKBest with the ANOVA F-test, keeping the 10 features most related to Survived.

Highest scores: Title_Mr (301.8), Sex_male (294.9), Title_Mrs (116.9), Pclass (82.0), Title_Miss (75.0).
Lowest scores: Title_Rare, Embarked_Q, FamilySize, and AgeGroup_Teen (all about 0).

Selected features: Pclass, Parch, Fare, IsAlone, FarePerPerson, Sex_male, Embarked_S, Title_Miss, Title_Mr, Title_Mrs.

## Results
Logistic Regression, 80/20 split, random_state=42:
- All 18 features: accuracy 0.810
- 10 selected features: accuracy 0.760
- 16 selected features: accuracy 0.810

## Conclusion
Title and sex were the strongest predictors, and the title feature I created ranked above the original columns. However, keeping only 10 features slightly lowered accuracy. The F-test scores each feature on its own, so it can drop features like Age that help only when combined with others. In this case, 16 features matched the full model, so selection helped to simplify the model but did not improve accuracy.
