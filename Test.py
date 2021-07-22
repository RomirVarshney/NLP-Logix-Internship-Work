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

def isPass(nfl):
    if(nfl["play_type"]) == "pass":
        return 1
    elif(nfl["play_type"]) == "run":
        return 0
    else:
        return None




nfl["yvar"] = nfl.apply(isPass, axis=1)
nfl= nfl[nfl["yvar"].notna()]
#print(nfl["yvar"].dtype)

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

nfl["ratio"]= nfl['score_differential'] / nfl['game_seconds_remaining'] 


features= ["yardline_100", "score_differential", "ydstogo", "game_seconds_remaining", "no_huddle", "down", "half_seconds_remaining", "quarter_seconds_remaining", "goal_to_go", "posteam_score", "defteam_score", "shotgun", "goal_to_go" , "posteam_timeouts_remaining", "defteam_timeouts_remaining", "total_home_score", "total_away_score"]
newFeatures = [
    "avg_pass_rate",
    'ratio'
]

nfl.replace([np.inf, -np.inf], np.nan, inplace = True)
nfl[features]= nfl[features].fillna(-999)


print(nfl.loc[nfl['game_seconds_remaining']== 0])

nfl["year"] = pd.to_datetime(nfl["game_date"], yearfirst=True, errors='coerce')
nfl["year"] = nfl["year"].dt.year
nfl_team_yards = nfl.groupby(["posteam", "year"])["yvar"].mean()

nfl = pd.merge(nfl, nfl_team_yards, how="left", on=["posteam", "year"])
nfl.rename(columns={"yvar_y": "avg_pass_rate"}, inplace=True)

#playtime = 'score_differential'/ 'game_seconds_remaining'
#fillFeatures= pd.DataFrame(nfl[features])
#fillFeatures= fillFeatures.dropna()

nfl["yvar"] = nfl.apply(isPass, axis=1)

features = features + newFeatures

x= nfl[features]
y= nfl["yvar"]

#print(nfl[features])
#print(nfl[features].dtypes)
#print(nfl[features].isnull().sum())
#print(x.shape)

train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=1)
model = RandomForestClassifier(n_estimators=175, max_depth=13, random_state=1)
model.fit(train_X,train_y)
predictions = model.predict_proba(test_X)

result = roc_auc_score(test_y,predictions[:, 1])
print("ROC AUC Score: " + str(result))

feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame(feature_importance, index=train_X.columns, columns=["Rate of Importance"])
feature_importance_df.sort_values("Rate of Importance")
print(feature_importance_df)

#accuracyScore= (metrics.accuracy_score(test_y, predictions))
#print(accuracyScore)

output = pd.DataFrame({'Predicted_play':predictions[:, 1], 'Actual_Play': test_y})
output.to_csv("predictions1.csv")
#print(predictions)
