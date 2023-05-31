import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score

def data_loading(PATH):
    df = pd.read_csv(PATH)
    array = df.to_numpy()
    return df, array

def tree_model_info(tree_model, feature_names, plot_tree=False):
    print(f"Number of depth: {tree_model.get_depth()}")
    print(f"Number of leaf node: {tree_model.get_n_leaves()}")
    print(f"Number of total node: {tree_model.tree_.node_count}")

    if plot_tree:
        plt.figure(figsize=(20, 20))
        tree.plot_tree(tree_model, feature_names=feature_names, filled=True)

def rf_model_info(rf_model, feature_names, plot_tree=False):
    model_num = np.random.choice(rf_model.get_params()["n_estimators"])
    sub_rf_model = rf_model[model_num]
    print(f"Random Forest Model#{model_num}")

    tree_model_info(sub_rf_model, feature_names, plot_tree=plot_tree)

def model_evaluation(model, X, y, print_dataset=False, print_metrics=True):
    y = y.ravel()

    y_preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_preds))
    r2 = r2_score(y, y_preds)

    if print_dataset:
        print(print_dataset)

    if print_metrics:
        print(f"RMSE: {rmse}")
        print(f"r2: {r2}")

    return rmse, r2

def tree_feature_importance(model, feature_names, print_top=10):
    df = pd.DataFrame(model.feature_importances_.reshape(-1,1), index=feature_names)
    df = df.rename(columns={0: "Feature Importance"}).sort_values(by="Feature Importance", ascending=False)
    print(df.head(print_top))

def plot_predictedVSreal(model, X, y, print_title=False):
    y_preds = model.predict(X)
    rmse, r2 = model_evaluation(model, X, y, print_dataset=False, print_metrics=False)

    plt.figure(figsize=(10, 7))
    plt.scatter(y, y_preds)
    plt.plot(y, y, color="red")
    plt.text(0, 95, f"RMSE: {round(rmse, 1)}, R2: {round(r2,2)}")
    plt.ylabel("Predicted CO2 conversion")
    plt.xlabel("Experimental CO2 conversion")
    if print_title:
        plt.title(print_title)

def plot_prediction(model, X_test, y_test, raw_data, exp_no=[478, 401, 392, 248]):
    raw_data = raw_data[raw_data["Train/Test Set"] == "Test"].reset_index()
    plt.figure(figsize=(17, 14))

    for i in range(len(exp_no)):
        exp_data = raw_data[raw_data["Experiment No"] == exp_no[i]]
        index = sorted(list(exp_data.index))

        y_preds = model.predict(X_test.to_numpy())

        temp = (X_test["Temperature (C)"].loc[index[0]:index[-1]]).to_numpy()
        y_preds = y_preds[index[0]:index[-1]+1]
        y_real = (y_test.loc[index[0]:index[-1]]).to_numpy()

        plt.subplot(2, int(round(len(exp_no)/2, 0)), i+1)
        if exp_no[i] == 392:
            plt.plot(temp[:3], y_preds[:3], label="Predicted")
            plt.scatter(temp[:3], y_real[:3], color="red", label="Real")
        
        else:
            plt.plot(temp, y_preds, label="Predicted")
            plt.scatter(temp, y_real, color="red", label="Real")

        plt.legend()
        plt.ylabel("CO2 Conversion %")
        plt.xlabel("Operation Temperature (C)")
        plt.title(f"Experiment No. {exp_no[i]}")
        plt.ylim(0,100)
