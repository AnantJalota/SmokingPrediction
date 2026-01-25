import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score


class SmokerPrediction():
    
    def __init__(self, df, drop):
        self.df = df
        self.drop = drop

    def load_and_preprocess(self):
        df = self.df.copy()
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        if 'gender' in df.columns:
            df['gender'] = (df['gender'] == 'M').astype(int)
        if 'oral' in df.columns:
            df['oral'] = (df['oral'] == 'Y').astype(int)
        if 'tartar' in df.columns:
            df['tartar'] = (df['tartar'] == 'Y').astype(int)
        if 'Cholesterol' in df.columns:
            df = df.rename(columns={'Cholesterol': 'cholesterol'})
        if self.drop:
            df = df.drop_duplicates(keep='first')
        return df

    def create_features(self):
        df = self.load_and_preprocess()
        # Core features
        df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
        df['waist_height_ratio'] = df['waist(cm)'] / df['height(cm)']
        df['avg_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
        df['eyesight_diff'] = np.abs(df['eyesight(left)'] - df['eyesight(right)'])
        df['hearing_diff'] = np.abs(df['hearing(left)'] - df['hearing(right)'])
        df['pulse_pressure'] = df['systolic'] - df['relaxation']

        # Cholesterol
        df['total_HDL_ratio'] = df['cholesterol'] / (df['HDL'] + 1)
        df['LDL_HDL_ratio'] = df['LDL'] / (df['HDL'] + 1)
        df['triglyceride_HDL_ratio'] = df['triglyceride'] / (df['HDL'] + 1)
        
        # Liver
        df['AST_ALT_ratio'] = df['AST'] / (df['ALT'] + 1)
        df['liver_enzyme_sum'] = df['AST'] + df['ALT'] + df['Gtp']
        
        # Interactions
        df['age_BMI'] = df['age'] * df['BMI']
        df['age_systolic'] = df['age'] * df['systolic']
        df['age_hemoglobin'] = df['age'] * df['hemoglobin']
        df['age_triglyceride'] = df['age'] * df['triglyceride']
        df['chol_BP'] = df['cholesterol'] * df['systolic']
        df['triglyceride_BP'] = df['triglyceride'] * df['systolic']
        df['triglyceride_glucose'] = df['triglyceride'] * df['fasting blood sugar']
        df['BMI_glucose'] = df['BMI'] * df['fasting blood sugar']
        df['waist_triglyceride'] = df['waist(cm)'] * df['triglyceride']
        df['BMI_cholesterol'] = df['BMI'] * df['cholesterol']
        df['hemoglobin_BP'] = df['hemoglobin'] * df['systolic']
        df['liver_glucose'] = df['liver_enzyme_sum'] * df['fasting blood sugar']
        df['AST_hemoglobin'] = df['AST'] * df['hemoglobin']
        
        # Flags
        df['hypertension'] = ((df['systolic'] >= 140) | (df['relaxation'] >= 90)).astype(int)
        df['high_cholesterol'] = (df['cholesterol'] >= 240).astype(int)
        df['low_HDL'] = (df['HDL'] < 40).astype(int)
        df['high_triglycerides'] = (df['triglyceride'] >= 200).astype(int)
        df['high_glucose'] = (df['fasting blood sugar'] >= 126).astype(int)
        df['prediabetes'] = ((df['fasting blood sugar'] >= 100) & (df['fasting blood sugar'] < 126)).astype(int)
        df['overweight'] = (df['BMI'] >= 25).astype(int)
        df['obese'] = (df['BMI'] >= 30).astype(int)
        df['metabolic_risk'] = (
            (df['waist(cm)'] > 90).astype(int) +
            (df['fasting blood sugar'] >= 100).astype(int) +
            (df['triglyceride'] >= 150).astype(int) +
            (df['HDL'] < 40).astype(int) +
            ((df['systolic'] >= 130) | (df['relaxation'] >= 85)).astype(int)
        )
        df['elevated_AST'] = (df['AST'] > 40).astype(int)
        df['elevated_ALT'] = (df['ALT'] > 40).astype(int)
        df['elevated_Gtp'] = (df['Gtp'] > 60).astype(int)
        df['kidney_concern'] = (df['serum creatinine'] > 1.2).astype(int)
        df['has_dental_issues'] = (df['dental caries'] > 0).astype(int)
        df['elevated_liver_enzymes'] = ((df['AST'] > 40) | (df['ALT'] > 40) | (df['Gtp'] > 60)).astype(int)
        df['central_obesity'] = (df['waist(cm)'] > 90).astype(int)   
        return df

    def decision_features(self):
        df = self.load_and_preprocess()
        
        df["BMI"] = df["weight(kg)"] / (df["height(cm)"]/100) ** 2
        df["GTP_BMI"] = df["Gtp"] * df["BMI"]
        df["TG_BMI"]  = df["triglyceride"] * df["BMI"]
        df["TG_weight"] = df["triglyceride"] / df["weight(kg)"]
        df["ht*hemoglobin"] = df["height(cm)"] * df["hemoglobin"]
        df["Gtp/LDL"] = df["Gtp"] / df["LDL"]
        df["Gtp/Cholesterol"] = df["Gtp"] / df["cholesterol"]
        df["HDL/Gtp"] = df["HDL"] / df["Gtp"]
        df["TG_HDL_ratio"] = df["triglyceride"] / df["HDL"]
        df["waist(cm)/height(cm)"] = df["waist(cm)"] / df["height(cm)"]
        df["Pressure"] = df["systolic"] * df["relaxation"]
        df["Cholesterol*triglyceride"] = df["cholesterol"] * df["triglyceride"]
        df["HDL/LDL"] = df["HDL"] / df["LDL"]
        df["AST/ALT"] = df["AST"] / df["ALT"]
        df["Gtp*height"] = df["Gtp"] * df["height(cm)"]
        return df

    def create_basic_features(self):
        df = self.load_and_preprocess()
        
        df["triglyceride"] = np.clip(df["triglyceride"], df["triglyceride"].quantile(0.01), df["triglyceride"].quantile(0.99))
        df["AST"] = np.clip(df["AST"], df["AST"].quantile(0.01),df["AST"].quantile(0.99))
        df["ALT"] = np.clip(df["ALT"], df["ALT"].quantile(0.01),df["ALT"].quantile(0.99))
        df["serum creatinine"] = np.clip(df["serum creatinine"], df["serum creatinine"].quantile(0.01),df["serum creatinine"].quantile(0.99))
        df["Gtp"] = np.clip(df["Gtp"], df["Gtp"].quantile(0.01),df["Gtp"].quantile(0.99))
        df["HDL"] = np.clip(df["HDL"], df["HDL"].quantile(0.01),df["HDL"].quantile(0.99))
        df["LDL"] = np.clip(df["LDL"], df["LDL"].quantile(0.01),df["LDL"].quantile(0.99))
        df["fasting blood sugar"] = np.clip(df["fasting blood sugar"], df["fasting blood sugar"].quantile(0.01),df["fasting blood sugar"].quantile(0.99))
        df["cholesterol"] = np.clip(df["cholesterol"], df["cholesterol"].quantile(0.01),df["cholesterol"].quantile(0.99))

        df = pd.get_dummies(df,columns=["gender", "tartar"],dtype=int,drop_first=True)
        df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)

        df['Gtp'] = np.log1p(df['Gtp'])
        df['AST'] = np.log1p(df['AST'])
        df['ALT'] = np.log1p(df['ALT'])
        df['triglyceride'] = np.log1p(df['triglyceride'])

        df.drop(["eyesight(left)", "eyesight(right)","hearing(left)", "hearing(right)","Urine protein"],axis=1,inplace=True)
        
        return df
    
    def scaling(self, data, scale, select, poly, k, model=None):
        
        X,y = data.drop(["smoking"],axis=1), data["smoking"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
        
        if model == LogisticRegression: 

            pipe = Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
            ('scale', StandardScaler())
            ])
            
            X_train = pipe.fit_transform(X_train, y_train)
            X_val = pipe.transform(X_val)
        
        else:
            
            if select:

                selector = SelectKBest(mutual_info_classif, k=k)
                selector.fit_transform(X_train, y_train)
                selected_features = X_train.columns[selector.get_support()].tolist()
                X_train = X_train[selected_features]
                X_val = X_val[selected_features]
            
            if scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val   = scaler.transform(X_val)
                        
        return X_train, X_val, y_train, y_val

        
    def train_xgboost_mcc_optimized(self, X_train, y_train, X_val, y_val, model):

        # Calculate class distribution
        n_nonsmokers = (y_train == 0).sum()
        n_smokers = (y_train == 1).sum()
        imbalance_ratio = n_nonsmokers / n_smokers

        best_overall_mcc = 0
        best_overall_model = None
        best_overall_threshold = 0.5
        best_overall_config = None
        
        if model == XGBClassifier:
        # XGBoost with this scale_pos_weight
            params_grid = {'scale_pos_weight':[imbalance_ratio],
                            'n_estimators': [300, 325], 
                            'max_depth': [5,6],
                            'learning_rate' : [0.05, 0.1],
                            'subsample': [0.8, 1],
                            'colsample_bytree': [0.8],
                            'colsample_bylevel': [0.95],
                            'gamma': [0.6, 1],
                            'reg_alpha' : [0.5, 1],
                            'reg_lambda': [1,2],
                            'min_child_weight':[1.5, 2]
                            }
            
            gs = GridSearchCV(XGBClassifier(random_state=42, n_jobs=-1,tree_method='hist',eval_metric='auc'), params_grid, cv=2, n_jobs=-1, scoring='matthews_corrcoef')
            gs.fit(X_train, y_train)            
            md = model(**gs.best_params_)
            md.fit(X_train, y_train)
            y_train_pred = md.predict(X_train)
            y_pred = md.predict(X_val)

        if (model == RandomForestClassifier) :
            params_grid = {'class_weight': [{0:1, 1: imbalance_ratio}],
                            'n_estimators': [300], 
                            'max_depth': [21,23],
                            'min_samples_split':[25,30],
                            'min_samples_leaf':[5,10],
                            'max_features':['sqrt'],
                            'max_samples':[0.75,1], 
                            'ccp_alpha':[0.000075]}                        
            
            gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), params_grid, cv=2, n_jobs=-1, scoring='matthews_corrcoef')
            gs.fit(X_train, y_train)
            md = model(**gs.best_params_)
            md.fit(X_train, y_train)
            y_train_pred = md.predict(X_train)
            y_pred = md.predict(X_val)
        
        for i in range(max(gs.cv_results_["rank_test_score"])):
            if f1_score(y_train, y_train_pred) - f1_score(y_val, y_pred) >= 0.11:
                params = pd.DataFrame(gs.cv_results_).sort_values(["rank_test_score"]).reset_index(drop=True).loc[i+1,'params']
                md = model(**params)
                md.fit(X_train, y_train)
                y_train_pred = md.predict(X_train)
                y_pred = md.predict(X_val)
                
                
        y_proba_val = md.predict_proba(X_val)[:, 1]
        
        # Optimize threshold for MCC
        best_mcc = 0
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in np.arange(0.30, 0.6, 0.05):
            y_train_threshold = (md.predict_proba(X_train)[:, 1] >= threshold).astype(int)
            y_pred_threshold = (y_proba_val >= threshold).astype(int)
            
            mcc = matthews_corrcoef(y_val, y_pred_threshold)
            precision = precision_score(y_val, y_pred_threshold)
            recall = recall_score(y_val, y_pred_threshold)
            f1 = f1_score(y_val, y_pred_threshold)
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
                best_metrics = (mcc, precision, recall, f1)
        
        # Track overall best
        if best_mcc > best_overall_mcc:
            best_overall_mcc = best_mcc
            best_overall_model = md
            best_overall_threshold = best_threshold
            
        # Final evaluation
        y_proba_val = best_overall_model.predict_proba(X_val)[:, 1]
        y_pred_optimized = (y_proba_val >= best_overall_threshold).astype(int)
        
        return best_overall_model, y_pred_optimized, best_overall_threshold


    def train_knn_robust(self,X_train, y_train, X_val, y_val, n_features=15):
        
        n_nonsmokers = (y_train == 0).sum()
        n_smokers = (y_train == 1).sum()
        imbalance_ratio = n_nonsmokers / n_smokers

        best_mcc = -1
        best_model = None
        best_threshold = 0.5
        best_k = 5
        best_metric = 'euclidean'
        
        # Test different K values and metrics
        k_values = [61,63,65,67,69]
        metrics = ['euclidean', 'manhattan']
        weights_options = ['uniform', 'distance']
        
        for k in k_values:
            for metric in metrics:
                for weights in weights_options:
                    
                    knn = KNeighborsClassifier(
                        n_neighbors=k,
                        weights=weights,
                        metric=metric,
                        n_jobs=-1
                    )
                    
                    knn.fit(X_train, y_train)
                    y_proba = knn.predict_proba(X_val)[:, 1]
                    
                    # Optimize threshold
                    for threshold in np.arange(0.30, 0.55, 0.05):
                        y_pred = (y_proba >= threshold).astype(int)
                        
                        mcc = matthews_corrcoef(y_val, y_pred)
                        precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
                        recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
                        
                        # Print only good results to reduce clutter
                        if mcc > best_mcc:
                            best_mcc = mcc
                            best_model = knn
                            best_threshold = threshold
                            best_k = k
                            best_metric = metric
                            best_weights = weights
                
        if best_model is None:
            return None, None, None
        
        y_proba_train = best_model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_proba_train >= best_threshold).astype(int)
        
        f1_train = f1_score(y_train, y_pred_train)

        y_proba_val = best_model.predict_proba(X_val)[:, 1]
        y_pred_val = (y_proba_val >= best_threshold).astype(int)
        
        f1_val = f1_score(y_val, y_pred_val, pos_label=1)
        
        gap = f1_train - f1_val
        
        if gap >= 0.11:
            return None, None, None
        
        return best_model, y_pred_val, threshold
    
    def pipeline_models(self, model, X_train, y_train, X_val, param_grid):
        
        if model in [LogisticRegression]:

            gs = GridSearchCV(LogisticRegression(), param_grid, cv=3, n_jobs=-1, scoring='matthews_corrcoef')
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_val)
        
        elif model == DecisionTreeClassifier:
            gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='matthews_corrcoef')
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_val)
        
        elif model == GaussianNB:
            gs = GridSearchCV(GaussianNB(priors=[0.4,0.6]), param_grid, cv=3, n_jobs=-1, scoring='matthews_corrcoef')            
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_val)
        
        return gs.best_params_, y_pred