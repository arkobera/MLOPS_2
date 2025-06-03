import xgboost
from sklearn.model_selection import KFold
import logging
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import root_mean_squared_error
import yaml
import dvclive
from dvclive import Live

# Create logs directory if it doesn't exist
log_dir = "C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/logs" 
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.DEBUG)
logger.addHandler(terminal_handler)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
terminal_handler.setFormatter(logger_formatter)
# Save log file in the logs directory
file_handler = logging.FileHandler(os.path.join(log_dir, 'modeling.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)



def load_parameters(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info("Parameters loaded successfully")
        print("Parameters loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        print(f"Error loading parameters: {e}")
        return {}
# src/modeling.py

class Trainer:
    def __init__(self,X,y,folds,test):

        self.X = X
        self.y = np.log1p(y)
        self.folds = folds
        self.test = test
        self.oofs = np.zeros(len(self.X))
        self.pred = np.zeros(len(self.test))
        
    def train(self):
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        for fold,(train_indx,test_indx) in enumerate(kf.split(self.X,self.y)):
            print(f"Fold : {fold+1}")
            x_train, y_train = self.X.iloc[train_indx],self.y.iloc[train_indx]
            x_valid, y_valid = self.X.iloc[test_indx], self.y.iloc[test_indx]

            self.model = xgboost.XGBRegressor(
                device='cuda',
                max_depth=8,
                colsample_bytree=0.9,
                subsample=0.9,
                n_estimators=1500,
                learning_rate=0.007,
                random_state=0,
                eval_metric="rmse"
            )
            self.model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],  
                verbose=300,
            )
            self.oofs[test_indx] = self.model.predict(x_valid)
            self.pred += self.model.predict(self.test)
            logger.info(f"Fold {fold+1} completed successfully")
        self.pred /= 5
        self.score = root_mean_squared_error(self.y, self.oofs)
        logger.info(f"Training completed with RMSE: {self.score}")

    def save_predictions(self,file_name='submission.csv'):
        try:
            os.makedirs("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Predictions/Submissions", exist_ok=True)
            submission = pd.DataFrame({
                'id': self.test.index,
                'Calories': np.expm1(self.pred)
            })
            submission.to_csv(os.path.join("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Predictions/Submissions", file_name), index=False)
            logger.info("Predictions saved successfully")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            print(f"Error saving predictions: {e}")

    def save_oofs(self, file_name='oofs.csv'):
        try:
            os.makedirs("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Predictions/Oofs", exist_ok=True)
            oofs = pd.DataFrame({
                'id': self.X.index,
                'Calories': np.expm1(self.oofs)
            })
            oofs.to_csv(os.path.join("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Predictions/Oofs", file_name), index=False)
            logger.info("OOFs saved successfully")
        except Exception as e:
            logger.error(f"Error saving OOFs: {e}")
            print(f"Error saving OOFs: {e}")

    def save_model(self, file_name='xgboost_model.pkl'):
        try:
            model = self.model
            os.makedirs("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Models/model", exist_ok=True)
            joblib.dump(model, os.path.join("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Models/model", file_name))
            logger.info("Model saved successfully as pkl")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            print(f"Error saving model: {e}")

    def save_score(self, file_name='score.txt'):
        try:
            os.makedirs("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Models/Evaluation", exist_ok=True)
            with open(os.path.join("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Models/Evaluation", file_name), 'w') as f:
                f.write(f"RMSE: {self.score}")
            logger.info("Score saved successfully")
        except Exception as e:
            logger.error(f"Error saving score: {e}")
            print(f"Error saving score: {e}")

    def load_model(self, file_name='xgboost_model.pkl'):
        try:
            model = joblib.load(os.path.join("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/Models", file_name))
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            return None
        
def main():
    try:
        params = load_parameters("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/params.yaml")
        train_data = pd.read_csv("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/data/raw/train.csv")
        test_data = pd.read_csv("C:/Users/ARKO BERA/Desktop/MLOPS/MLOPS_2/data/raw/test.csv")
        logger.info("Data loaded successfully")
        X = train_data.copy()
        y = X.pop('Calories')
        test = test_data.copy()
        trainer = Trainer(X, y, folds=3, test=test)
        trainer.train()
        trainer.save_predictions()
        trainer.save_oofs()
        trainer.save_score()
        trainer.save_model()
        metrics = trainer.score
        with Live(save_dvc_exp=True) as live:
            live.log_metric('RMSLE', metrics)
            live.log_params(params)
        logger.info("Training and saving process completed successfully")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"File not found: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return
    
if __name__ == "__main__":
    main()
    logger.info("Modeling script executed successfully")
    print("Modeling script executed successfully")
    
