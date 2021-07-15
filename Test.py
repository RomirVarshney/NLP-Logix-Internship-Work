import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import numpy as np

nfl = pd.read_csv("NFL Play by Play 2009-2018 (v5).csv", low_memory=False)


def isPass(nfl):
    if(nfl["play_type"]) == "pass":
        return 1
    elif(nfl["play_type"]) == "run":
        return 0
    else:
        return None


nfl["yvar"] = nfl.apply(isPass, axis=1)
nfl= nfl[nfl["yvar"].notna()]
#droppedy= pd.DataFrame(nfl["yvar"])
#droppedy= droppedy.dropna()
nfl.head()
nfl= nfl.fillna(-999)
features= ["yardline_100", "score_differential", "ydstogo", "game_seconds_remaining", "no_huddle", "down"]
#fillFeatures= pd.DataFrame(nfl[features])
#fillFeatures= fillFeatures.dropna()

x= nfl[features]
y= nfl["yvar"]

print(nfl[features])
print(nfl[features].dtypes)
print(nfl[features].isnull().sum())
print(x.shape)

train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_X,train_y)
predictions = model.predict_proba(test_X)


output = pd.DataFrame({'Predicted_play':predictions[:, 1], 'Actual_Play': test_y})
output.to_csv("predictions.csv")


# Notable Features: Yardline, Score Differential, Yards to Go, Game Seconds Remaining, Huddle/No Huddle, Down, Posteam (Categorical), Defteam (Categorical), 