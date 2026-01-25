import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from model import SmokerPrediction

df = pd.read_csv(r"smoking.csv")

##Logistic Regression
sp = SmokerPrediction(df, True)

df3 = sp.create_basic_features()

X_train, X_val, y_train, y_val = sp.scaling(df3, True, True, True, 125, LogisticRegression)

params_grid = {
    'penalty':['l1','l2'],
    'solver':['saga','lbfgs'],
    'C':[0.005,0.01]
}

params, y_val_reg = sp.pipeline_models(LogisticRegression, X_train, y_train, X_val, params_grid)

##DecisionTreeClassifier
sp = SmokerPrediction(df, True)

print("\nPreprocessing...")

df1 = sp.decision_features()

X_train, X_val, y_train, y_val = sp.scaling(df1, False, True, False, 32, DecisionTreeClassifier)

params_grid = {
    'criterion':['entropy'],
    'max_depth':list(range(9,11,2)),
    'min_samples_leaf':[60, 80],
    'min_samples_split':[80,100],
    'class_weight':["balanced"],
}
params2, y_pred_dt = sp.pipeline_models(DecisionTreeClassifier, X_train, y_train, X_val, params_grid)

##KNearestNeighbours
sp = SmokerPrediction(df, True)

df2 = sp.load_and_preprocess()

df2.drop(["systolic", "tartar"], axis=1, inplace=True)

X_train, X_val, y_train, y_val = sp.scaling(df2, True, True, False, 15, KNeighborsClassifier)

knn_model, y_pred_knn, threshold = sp.train_knn_robust(X_train, y_train, X_val, y_val, n_features=15)

##Naive Bayes
sp = SmokerPrediction(df, True)

df4 = sp.create_basic_features()

df4 = df4[["age",'gender_1',"tartar_1", "dental caries","Gtp","BMI","triglyceride","LDL","relaxation","smoking"]]

X_train, X_val, y_train, y_val = sp.scaling(df4, True, False, False, 0, GaussianNB)

params, y_val_1 = sp.pipeline_models(GaussianNB, X_train, y_train, X_val, param_grid={"priors":[[0.4,0.6]]})

#RandomForestClassifier
sp1 = SmokerPrediction(df, False)

df5 = sp1.create_features()

X_train, X_val, y_train, y_val = sp1.scaling(df5, False, False, False, 0, RandomForestClassifier)

model, y_pred_rf, threshold = sp1.train_xgboost_mcc_optimized(X_train, y_train, X_val, y_val, RandomForestClassifier)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

y_pred_rf = (model.predict_proba(X_val) >= threshold)[:,1].astype(int)

##XGBClassifier
sp1 = SmokerPrediction(df, False)

df6 = sp1.create_features()

X_train, X_val, y_train, y_val = sp1.scaling(df6, False, False, False, 0, XGBClassifier)

model1, y_pred_xgb, threshold = sp1.train_xgboost_mcc_optimized(X_train, y_train, X_val, y_val, XGBClassifier)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

y_pred_xgb = (model.predict_proba(X_val) >= threshold)[:,1].astype(int)

#Best Models.
#Logistic Regression: {'C': 0.01, 'penalty': 'l2', 'solver': 'saga', 'degree': 2, 'k': 125}
#Decision Tree: 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 9, 'min_samples_leaf': 80, 'min_samples_split': 80
#Naive Bayes: simple model
#KNN: {'metric': manhattan, 'weights': uniform, 'threshold': 0.350}
#RandomForestClassifier: {'n_estimators': 275, 'max_depth': 23,'min_samples_split':30,'min_samples_leaf':10,'max_features':'sqrt','random_state':42,'n_jobs':-1,'max_samples':0.75, 'ccp_alpha':0.000075}
#XGBClassifier: {'n_estimators': 325, 'max_depth': 6,'learning_rate' : 0.1,'subsample': 0.8,'colsample_bytree': 0.8,'colsample_bylevel': 0.95,'gamma': 0.6,'reg_alpha' : 0.5,'reg_lambda': 2,'random_state':42,'tree_method':'hist','eval_metric':'auc','use_label_encoder':False, 'min_child_weight':1.5}

#Creating pickles

#XGBClassifier
df = pd.read_csv("Smoking_train.csv")

df_feat_xgb = SmokerPrediction(df, drop=False).create_features()

X = df_feat_xgb.drop("smoking", axis=1)
y = df_feat_xgb["smoking"]

final_model = XGBClassifier(n_estimators = 325, max_depth = 6, learning_rate = 0.1, subsample = 0.8, colsample_bytree = 0.8, colsample_bylevel = 0.95, gamma = 0.6, reg_alpha = 0.5, reg_lambda = 2, random_state = 42, tree_method ='hist', eval_metric = 'auc', use_label_encoder = False, min_child_weight = 1.5)

final_model.fit(X, y)

artifact = {
    "model": final_model,
    "features": X.columns.tolist(),
    "threshold": 0.5
}

with open("smoker_model_xgb.pkl", "wb") as f:
    pickle.dump(artifact, f)

#RandomForest
df_feat_rf = SmokerPrediction(df, drop=False).create_features()

X = df_feat_rf.drop("smoking", axis=1)
y = df_feat_rf["smoking"]

# 3. Train ONE final model (best params you found)
final_model = RandomForestClassifier(n_estimators = 275, max_depth = 23, min_samples_split = 30, min_samples_leaf = 10, max_features = 'sqrt', random_state = 42, n_jobs = -1, max_samples = 0.75, ccp_alpha = 0.000075)

final_model.fit(X, y)

# 4. Save EVERYTHING needed for inference
artifact = {
    "model": final_model,
    "features": X.columns.tolist(),
    "threshold": 0.5
}

with open("smoker_model_rf.pkl", "wb") as f:
    pickle.dump(artifact, f)
    
#LogisticRegression
df_lr_feat = SmokerPrediction(df, drop=True).create_basic_features()

X = df_lr_feat.drop("smoking", axis=1)
y = df_lr_feat["smoking"]

lr_model = Pipeline([('poly', PolynomialFeatures(degree=2)), 
                    ('select',SelectKBest(score_func=mutual_info_classif, k=125)),
                    ('scale',StandardScaler()),
                    ('model',LogisticRegression(penalty='l2', C=0.01, solver='saga'))])

lr_model.fit(X, y)

artifact = {
    "model": lr_model,
    "threshold": 0.5
}

with open("smoker_model_lr.pkl", "wb") as f:
    pickle.dump(artifact, f)
    
#DecisionTreeClassifier
    
df_dt_feat = SmokerPrediction(df, drop=True).decision_features()

X,y = df_dt_feat.drop(["smoking"], axis=1), df_dt_feat["smoking"]

dt_model = DecisionTreeClassifier(class_weight = 'balanced', criterion = 'entropy', max_depth = 9, min_samples_leaf = 80, min_samples_split = 80,random_state=42)

dt_model.fit(X,y)

artifact = {"model":dt_model,
            "columns":X.columns.tolist()
            "threshold":0.5}

with open("smoker_model_dt.pkl", "wb") as f:
    pickle.dump(artifact,f)
    
#NaiveBayes

df_nb_feat = SmokerPrediction(df, drop=True).create_basic_features()

cols = ["age",'gender_1',"tartar_1", "dental caries","Gtp","BMI","triglyceride","LDL","relaxation","smoking"]

df_nb_feat = df_nb_feat[cols]

X,y = df_nb_feat.drop(["smoking"],axis=1), df_nb_feat["smoking"]

nb = Pipeline([('scale', StandardScaler()), ('model', GaussianNB(priors=[0.4,0.6]))])

nb.fit(X,y)

artifact = {"model":nb,
            "features":cols,
            "threshold":0.5}

with open("smoker_model_nb.pkl", "wb") as f:
    pickle.dump(artifact,f)
    
#KNN

df_knn_feat = SmokerPrediction(df, drop=True).load_and_preprocess()

cols = ["gender", "age", "height(cm)", "weight(kg)", "waist(cm)", "eyesight(left)", "eyesight(right)", "hearing(left)", "relaxation", "fasting blood sugar", "triglyceride", "HDL", "LDL", "hemoglobin", "serum creatinine", "AST", "ALT", "Gtp", "oral", "dental caries", "smoking"]

df_knn_feat = df_knn_feat[cols]

X,y = df_knn_feat.drop(["smoking"], axis=1), df_knn_feat["smoking"]

knn = KNeighborsClassifier(n_neighbors = 65, metric = 'manhattan', weights = 'uniform', n_jobs=-1)

knn.fit(X,y)

artifact = {"model":knn,
            "features":cols,
            "threshold":0.35}

with open("smoker_model_knn.pkl", "wb") as f:
    pickle.dump(artifact,f)