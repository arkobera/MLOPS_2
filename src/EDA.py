import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from warnings import filterwarnings

filterwarnings("ignore")

import logging 
import os

# Create logs directory if it doesn't exist
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.DEBUG)
logger.addHandler(terminal_handler)

logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
terminal_handler.setFormatter(logger_formatter)

# Save log file in the logs directory
file_handler = logging.FileHandler(os.path.join(log_dir, 'EDA.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)


class Load_Data:
    def __init__(self, file_path=None,file_df=None):
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        elif file_df is not None:
            self.df = file_df
        #self.id = self.df['id']
        #self.df.drop(columns=['id'], inplace=True)
        print("Shape = ", self.df.shape)

    def summarize(self, include='all'):
        print("=" * 50, 'SUMMARY', '=' * 50)
        if include == 'numerical':
            summarize_df = self.df.describe(include=['number']).T
        elif include == 'categorical':
            summarize_df = self.df.describe(include=['object', 'category']).T
        else:
            summarize_df = self.df.describe(include='all').T

        summarize_df['dtype'] = self.df.dtypes
        summarize_df['missing'] = self.df.isnull().sum()
        summarize_df['unique'] = self.df.nunique()
        summarize_df['duplicates'] = self.df.duplicated().sum()
        summarize_df['most_frequent'] = self.df.select_dtypes(include=['object', 'category']).apply(
            lambda col: col.value_counts().idxmax() if col.nunique() > 0 else None
        )

        def highlight(val):
            if isinstance(val, (int, float)):
                if val > 100000:
                    return 'background-color: red'
                elif val > 50000:
                    return 'background-color: orange'
                elif val > 10000:
                    return 'background-color: blue'
                elif val < 1000:
                    return 'background-color: green'
            return ''

        summarize_df.drop(columns=['25%', '50%', '75%', 'count', 'most_frequent'], inplace=True)
        styled_df = summarize_df.style.applymap(highlight, subset=['missing', 'unique'])
        return styled_df

    def visualize(self, include='all', sample=10000, exclude=[], target=None):
        sample_df = self.df.sample(sample)
    
        if include == 'numerical':
            columns_to_plot = self.df.select_dtypes(include=['number']).columns
        elif include == 'categorical':
            columns_to_plot = self.df.select_dtypes(include=['object', 'category']).columns
        else:
            columns_to_plot = self.df.columns
    
        columns_to_plot = [col for col in columns_to_plot if col not in exclude]
    
        if 'numerical' in include or 'all' in include:
            print("=" * 50, 'Visualizing Numerical Features', '=' * 50)
            numerical_cols = self.df.select_dtypes(include=['number']).columns
            for col in numerical_cols:
                if col not in exclude:
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=(f'{col} - Histogram', f'{col} - Boxplot', f'{col} vs {target}'),
                        column_widths=[0.33, 0.33, 0.33]
                    )
                
                    hist = px.histogram(sample_df, x=col, title=f'{col} - Histogram')
                    fig.add_trace(hist.data[0], row=1, col=1)
                
                    box = px.box(sample_df, y=col, title=f'{col} - Boxplot')
                    fig.add_trace(box.data[0], row=1, col=2)
                
                    if target and target in self.df.columns:
                        box_target = px.box(sample_df, x=target, y=col, title=f'{col} vs {target}')
                        fig.add_trace(box_target.data[0], row=1, col=3)
                
                    fig.update_layout(title=f'{col} - Distribution and Feature Dependence', showlegend=False)
                    fig.show()
    
        if 'categorical' in include or 'all' in include:
            print("=" * 50, 'Visualizing Categorical Features', '=' * 50)
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in exclude:
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=(f'{col} - Count Plot', f'{col} - Pie Chart', f'{col} vs {target}'),
                        column_widths=[0.33, 0.33, 0.33],
                        specs=[[{"type": "bar"}, {"type": "pie"}, {"type": "bar"}]]
                    )
                
                    count_data = sample_df[col].value_counts().reset_index()
                    count_data.columns = [col, 'count']
                    count = px.bar(count_data, x=col, y='count', title=f'{col} - Count Plot')
                    fig.add_trace(count.data[0], row=1, col=1)
                
                    pie = px.pie(sample_df, names=col, title=f'{col} - Pie Chart')
                    fig.add_trace(pie.data[0], row=1, col=2)
                
                    if target and target in self.df.columns:
                        category_mean = sample_df.groupby(col)[target].mean().reset_index()
                        cat_target = px.bar(category_mean, x=col, y=target, title=f'{col} vs {target}')
                        fig.add_trace(cat_target.data[0], row=1, col=3)
                
                    fig.update_layout(title=f'{col} - Distribution and Feature Dependence', showlegend=True)
                    fig.show()


    def impute_columns(self, strategies=None, constant_values=None):
        if strategies is None:
            print("No strategies provided. Automatically imputing missing values.")
            strategies = {}

            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    strategies[col] = 'mean'
                else:
                    strategies[col] = 'mode'

        for col, strategy in strategies.items():
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in the dataframe. Skipping.")
                continue

            if strategy == 'mean':
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f"Imputed '{col}' with mean.")
                else:
                    print(f"Skipping '{col}' as it's not numeric for mean imputation.")

            elif strategy == 'median':
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Imputed '{col}' with median.")
                else:
                    print(f"Skipping '{col}' as it's not numeric for median imputation.")

            elif strategy == 'mode':
                if not self.df[col].isnull().all():
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
                    print(f"Imputed '{col}' with mode.")
                else:
                    print(f"Cannot compute mode for '{col}' as all values are NaN.")

            elif strategy == 'constant':
                if constant_values and col in constant_values:
                    self.df[col].fillna(constant_values[col], inplace=True)
                    print(f"Imputed '{col}' with constant value '{constant_values[col]}'.")
                else:
                    print(f"Error: Provide a constant value for column '{col}' in 'constant_values' dictionary.")

            else:
                print(f"Error: Unsupported strategy '{strategy}' for column '{col}'. Use 'mean', 'median', 'mode', or 'constant'.")

    def feature_target_dependence(self, target_col, exclude=[]):
        """
        Analyze the dependence of the target column on other features.

        Parameters:
        - target_col (str): The target column name.
        - exclude (list): List of feature column names to exclude from analysis.

        Returns:
        - pd.DataFrame: Summary of the dependence analysis.
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
        dependence_summary = []

        for col in self.df.columns:
            if col == target_col or col in exclude:
                continue
            
            # Numerical target
            if pd.api.types.is_numeric_dtype(self.df[target_col]):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Drop rows with NaN in either column
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data) > 1:
                        stat, p_value = stats.pearsonr(valid_data[col], valid_data[target_col])
                        dependence_summary.append([col, 'numerical', 'Pearson Correlation', stat, p_value])
                    else:
                        dependence_summary.append([col, 'numerical', 'Pearson Correlation', 'Insufficient Data', 'N/A'])
                elif pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                    # Perform ANOVA
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data[col].unique()) > 1 and len(valid_data) > 1:
                        groups = [valid_data[valid_data[col] == level][target_col] for level in valid_data[col].unique()]
                        stat, p_value = stats.f_oneway(*groups)
                        dependence_summary.append([col, 'categorical', 'ANOVA', stat, p_value])
                    else:
                        dependence_summary.append([col, 'categorical', 'ANOVA', 'Insufficient Data', 'N/A'])

            # Categorical target
            elif pd.api.types.is_categorical_dtype(self.df[target_col]) or pd.api.types.is_object_dtype(self.df[target_col]):
                if pd.api.types.is_categorical_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                    # Perform Chi-Square Test
                    contingency_table = pd.crosstab(self.df[col], self.df[target_col])
                    if contingency_table.size > 1:
                        stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                        dependence_summary.append([col, 'categorical', 'Chi-Square Test', stat, p_value])
                    else:
                        dependence_summary.append([col, 'categorical', 'Chi-Square Test', 'Insufficient Data', 'N/A'])
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    # Perform ANOVA
                    valid_data = self.df[[col, target_col]].dropna()
                    if len(valid_data[target_col].unique()) > 1 and len(valid_data) > 1:
                        groups = [valid_data[valid_data[target_col] == level][col] for level in valid_data[target_col].unique()]
                        stat, p_value = stats.f_oneway(*groups)
                        dependence_summary.append([col, 'numerical', 'ANOVA', stat, p_value])
                    else:
                        dependence_summary.append([col, 'numerical', 'ANOVA', 'Insufficient Data', 'N/A'])

        df =  pd.DataFrame(dependence_summary, columns=['Feature', 'Feature Type', 'Test Used', 'Statistic', 'p-value'])
        df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')  # Convert to numeric, set errors to NaN if not possible
        df['p-value'] = df['p-value'].apply(lambda x: round(x, 5) if pd.notnull(x) else x)  # Round only if not NaN
        #df['p-value'] = df['p-value'].apply(lambda x:round(x,5))
        def highlight(val):
            if isinstance(val, (int, float)):
                if val <0.05:
                    return 'background-color:green'     
            return ''

        dep_df = df.style.applymap(highlight, subset=['p-value'])
        return dep_df
        
    def get_df(self):
        return self.df
    
def main():
    try:
        os.makedirs("../data/EDA", exist_ok=True)  # Ensure the directory exists
        train_df = pd.read_csv("../data/raw/train.csv")
        eda = Load_Data(file_df=train_df)
        summary = eda.summarize(include='all')
        summary.to_excel("../data/EDA/EDA_summary.xlsx", index=False)
        logger.info("EDA summary saved to Excel.")
        #eda.visualize(include='all', sample=1000, target='Calories')
        dep_df = eda.feature_target_dependence(target_col='Calories')
        dep_df.to_excel("../data/EDA/EDA_dependence.xlsx", index=False)  # Save dependence DataFrame to Excel
        logger.info("EDA dependence analysis saved to Excel.")
        #print("EDA completed successfully.")
    except Exception as e:
        logger.error(f"Error occurred in EDA: {e}")
        print(f"Error occurred in EDA: {e}")

if __name__ == "__main__":
    main()
    logger.info("EDA completed successfully.")
    print("EDA completed successfully.")