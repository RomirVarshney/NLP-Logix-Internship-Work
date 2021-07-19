from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#import numpy as np

nfl = pd.read_csv("NFL Play by Play 2009-2018 (v5).csv", low_memory=False)

print()

# Combining Posteam Values to Updated
nfl["posteam"].replace("JAC", "JAX", inplace = True)
nfl["posteam"].replace("STL", "LA", inplace = True)
nfl["posteam"].replace("SD", "LAC", inplace = True)

# Grouping Posteam Values to Change to Numeric Average
#nflNew = nfl.groupby(by = "posteam")
#nflNew["passPercentage"].nflNew["yvar"].mean()
nfl.groupby(["posteam"])["yval"].sum()
print(nfl["sum"].head())

print(nfl["play_type"].value_counts(ascending = True))


# Changing Quantitative Play Type Values to Boolean
def isPass(nfl):
    if(nfl["play_type"]) == "pass":
        return 1
    elif(nfl["play_type"]) == "run":
        return 0
    else:
        return None

# Dropping the Null Values of the y-variable
nfl["yvar"] = nfl.apply(isPass, axis=1)
nfl= nfl[nfl["yvar"].notna()]

# Filling Null Values of Features with -999
nfl= nfl.fillna(-999)
features= ["yardline_100", "score_differential", "ydstogo", "game_seconds_remaining", "no_huddle", "down"]


x= nfl[features]
y= nfl["yvar"]

#print(nfl[features])
#print(nfl[features].dtypes)
#print(nfl[features].isnull().sum())
#print(x.shape)

# Creating the Machine Learning Model
train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X,train_y)
predictions = model.predict_proba(test_X)

# Outputting the Results to a CSV File
output = pd.DataFrame({'Predicted_play':predictions[:, 1], 'Actual_Play': test_y})
output.to_csv("predictions.csv")

# Outputting Feature Importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame(feature_importance, index = train_X.columns, columns = ["Rate of Importance"])
print(feature_importance_df)

# Outputting the ROC AUC Score
print()
result = roc_auc_score(test_y, predictions[:, 1])
print("ROC AUC Score: " + str(result))
print()
{"mode":"full","isActive":false}