import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.base import clone

def get_metrics(model, X_train, y_train, X_test, y_test, FR_train_time, FR_test_time):
    
    #training time = fit time + feature reduction train time
    start_train=time.time()
    model.fit(X_train, y_train)
    total_train_time= time.time() - start_train + FR_train_time

    #inference time = predict time + feature reduction test time
    start_test = time.time()
    y_pred = model.predict(X_test)
    total_infer_time = time.time() - start_test + FR_test_time
    #convert to Microseconds per Sample
    n_samples = len(y_test)
    inference_us_per_sample = (total_infer_time / n_samples) * 1_000_000 
    
    #precision, recall, f1-score
    p = precision_score(y_test, y_pred, average='weighted') * 100
    r = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100

    return [p, r, f1, total_train_time, inference_us_per_sample]


if __name__ == "__main__":
    train_df = pd.read_csv('Data/UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('Data/UNSW_NB15_testing-set.csv')

    #1 preprocess data
    #1.1: drop "id" column
    train_df.drop(columns=["id"], inplace=True, errors='ignore')
    test_df.drop(columns=["id"], inplace=True, errors='ignore')

    #1.2: Fill null values in "service" col
    train_df['service'] = train_df['service'].replace('-', 'other').fillna('other')
    test_df['service'] = test_df['service'].replace('-', 'other').fillna('other')

    #1.3: One-hot encoding proto, service, state
    nominal_features = ['proto', 'service', 'state']
    train_df = pd.get_dummies(train_df, columns=nominal_features)
    test_df = pd.get_dummies(test_df, columns=nominal_features)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
    print(train_df.shape)
    print(test_df.shape)

    #1.4: Split features and labels
    y_train_binary = train_df['label']
    y_test_binary = test_df['label']
    y_train_multiclass = train_df['attack_cat']
    y_test_multiclass = test_df['attack_cat']

    le = LabelEncoder()
    y_train_multiclass = le.fit_transform(y_train_multiclass)
    y_test_multiclass = le.transform(y_test_multiclass)

    X_train_base = train_df.drop(columns = ['label', 'attack_cat'])
    X_test_base = test_df.drop(columns = ['label', 'attack_cat'])
    print(X_train_base.shape)


    #2 Feature reduction 
    k=4 #number of features to keep

    # 2.1: Feature Selection (correlation matrix) 
    start_select_time_train = time.time()                  #to calculate timeFR_train
    correlation_matrix = X_train_base.corr()
    C_i = correlation_matrix.mean()
    selected_features = C_i.sort_values(ascending=False).head(k).index.tolist()
    X_train_sel = X_train_base[selected_features]
    FS_train_time = time.time() - start_select_time_train  #feature selection train time

    start_select_time_test = time.time()                   #to calculate timeFR_test
    X_test_sel = X_test_base[selected_features]
    FS_test_time = time.time() - start_select_time_test    #feature selection time

    print(selected_features)
    print(X_train_sel)

    # 2.2: Feature Extraction (PCA)
    scaler = MinMaxScaler()
    pca= PCA(n_components=k)

    start_extract_time_train = time.time(); #to calculate timeFR_train
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_train_ext = pca.fit_transform(X_train_scaled)
    FE_train_time = time.time() - start_extract_time_train 

    start_extract_time_test = time.time(); #to calculate timeFR_test
    X_test_scaled = scaler.transform(X_test_base) 
    X_test_ext = pca.transform(X_test_scaled)
    FE_test_time = time.time() - start_extract_time_test 
    print(X_train_ext)

    #3 Classifiers
    models = {
    "Decision Tree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(max_depth=5),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(max_iter=100, hidden_layer_sizes=200),
    "Naive Bayes": BernoulliNB()
}
    #4.1: Display binary classification results
    results_data = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Feature Extraction results 
        model_ext = clone(model)
        ext_metrics = get_metrics(model_ext, X_train_ext, y_train_binary, X_test_ext, y_test_binary, FE_train_time, FE_test_time)
        
        # Feature Selection results 
        model_sel = clone(model)
        sel_metrics = get_metrics(model_sel, X_train_sel, y_train_binary, X_test_sel, y_test_binary, FS_train_time, FS_test_time)
        
        row = [name] + ext_metrics + sel_metrics
        results_data.append(row)

    # Format dataframe
    columns = pd.MultiIndex.from_product(
        [["Feature Extraction", "Feature Selection"], 
        ["P", "R", "F1", "Training (s)", "Inference (us)"]]
    )
    df_results = pd.DataFrame(
        [r[1:] for r in results_data], 
        index=[r[0] for r in results_data], 
        columns=columns
    )
    print(df_results.round(2))
    print("4 selected/extracted features and binary classification results")  


    #4.2: Display multi-class classification results
    results_data_1 = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Feature Extraction results 
        model_ext = clone(model)
        ext_metrics = get_metrics(model_ext, X_train_ext, y_train_multiclass, X_test_ext, y_test_multiclass, FE_train_time, FE_test_time)
        
        # Feature Selection results 
        model_sel = clone(model)
        sel_metrics = get_metrics(model_sel, X_train_sel, y_train_multiclass, X_test_sel, y_test_multiclass, FS_train_time, FS_test_time)
        
        row = [name] + ext_metrics + sel_metrics
        results_data_1.append(row)

    # Format dataframe
    columns = pd.MultiIndex.from_product(
        [["Feature Extraction", "Feature Selection"], 
        ["P", "R", "F1", "Training (s)", "Inference (us)"]]
    )

    df_results = pd.DataFrame(
        [r[1:] for r in results_data_1], 
        index=[r[0] for r in results_data_1], 
        columns=columns
    )

    print(df_results.round(2))
    print("4 selected/extracted features and multi-class classification results")