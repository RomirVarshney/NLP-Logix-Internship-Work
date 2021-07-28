from numpy import average
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from numpy.core.fromnumeric import mean
import numpy as np
nfl = pd.read_csv("NFL Play by Play 2009-2018 (v5).csv", low_memory=False)

#print("COLUMN NAMES:")
#for col_name in nfl.columns:
 #   print(col_name)
#print("\n")

# Converting Categorical Play Type Values to Numeric
def isPass(nfl):
    if(nfl["play_type"]) == "pass":
        return 1
    elif(nfl["play_type"]) == "run":
        return 0
    else:
        return None



# Creating y-variable
nfl["yvar"] = nfl.apply(isPass, axis=1)
nfl= nfl[nfl["yvar"].notna()]
#print(nfl["yvar"].dtype)

# Combining Posteam Variables to Correct Values
nfl["posteam"].replace("JAC", "JAX", inplace =True)
nfl["posteam"].replace("STL", "LA", inplace =True)
nfl["posteam"].replace("SD", "LAC", inplace =True)


#nflNew= nfl.groupby(by="posteam")
#nfl["passProportion"]= nflNew["yvar"].mean()
#print(nfl["passProportion"].head())

#nfl["sum"]=nfl.groupby(["posteam"])
#print(nfl["sum"].head())

#def passProportion(nfl):
#print(nflNew["yvar"].head)


#droppedy= pd.DataFrame(nfl["yvar"])
#droppedy= droppedy.dropna()
#nfl.head()

# Creating the Ratio Features with Score Differential and Game Seconds
nfl["ratio"] = nfl['score_differential'] / nfl['game_seconds_remaining']

# All Original Features
features = [
"yardline_100",
"score_differential",
"ydstogo",
"game_seconds_remaining",
"no_huddle",
"down",
"half_seconds_remaining",
"quarter_seconds_remaining",
"posteam_score",
"defteam_score",
"shotgun",
"goal_to_go",
"posteam_timeouts_remaining",
"defteam_timeouts_remaining",
"total_home_score",
"total_away_score"
]

# All Newly Created Features
newFeatures = [
"avg_pass_rate",
'ratio',
]

# Filling in Feature Null Values with -999
nfl.replace([np.inf, -np.inf], np.nan, inplace = True)
nfl[features]= nfl[features].fillna(-999)



#print(nfl.loc[nfl['game_seconds_remaining']== 0])

# Seperating Data (Pass Percentage) by Posteam and Year
nfl["year"] = pd.to_datetime(nfl["game_date"], yearfirst=True, errors='coerce')
nfl["year"] = nfl["year"].dt.year
nfl_team_yards = nfl.groupby(["posteam", "year"])["yvar"].mean()

nfl = pd.merge(nfl, nfl_team_yards, how="left", on=["posteam", "year"])
nfl.rename(columns={"yvar_y": "avg_pass_rate"}, inplace=True)

# Dropping Ratio Null Values (Dividing by Zero)
nfl["ratio"]= nfl["ratio"].fillna(-999)

#playtime = 'score_differential'/ 'game_seconds_remaining'
#fillFeatures= pd.DataFrame(nfl[features])
#fillFeatures= fillFeatures.dropna()

# Adding y-variable to Data Set
nfl["yvar"] = nfl.apply(isPass, axis=1)

# Combining Features
features = features + newFeatures

# Creating X and Y Variables
x = nfl[features]
y = nfl["yvar"]

#print(nfl[features])
#print(nfl[features].dtypes)
#print(nfl[features].isnull().sum())
#print(x.shape)

# Creating the Train/Test Split, and Random Forest Classifier
train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
model = RandomForestClassifier(n_estimators=175, max_depth=13, random_state=1)
model.fit(train_X,train_y)
predictions = model.predict_proba(test_X)

# Printing ROC AUC Score
print()
result = roc_auc_score(test_y,predictions[:, 1])
print("ROC AUC Score: " + str(result))
print()

# Creating, Sorting, and Printing Feature Importance
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame(feature_importance, index=train_X.columns, columns=["Rate of Importance"])
print(feature_importance_df.sort_values("Rate of Importance", ascending = False))
print()
#print(feature_importance_df.sort_values)

#accuracyScore= (metrics.accuracy_score(test_y, predictions))
#print(accuracyScore)

# Outputting Predictions and Actual Play to CSV File
output = pd.DataFrame({'Predicted_play':predictions[:, 1], 'Actual_Play': test_y})
output.to_csv("predictions1.csv")
#print(predictions)