import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.compose._column_transformer")
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
from data_processor import DataProcessor  
from pathlib import Path


tabpfn_cache_dir = Path(r"D:\workspace\TabPFN\tabpfn")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(tabpfn_cache_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = str(tabpfn_cache_dir / "tabpfn-v2.5-regressor-v2.5_real.ckpt")

def load_data(folder_path, y_path=None, is_train=True):

    folder = Path(folder_path)

    csv_files = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".csv"
    ]

    csv_files.sort(
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    spectra = []
    sample_ids = []

    for csv_file in csv_files:
        data = pd.read_csv(csv_file, header=None)

        spectral_data = pd.to_numeric(
            data.iloc[:, 1],
            errors="raise"
        ).to_numpy(dtype=np.float32)

        spectra.append(spectral_data)
        sample_ids.append(csv_file.name)

    X = np.asarray(spectra, dtype=np.float32)
    sample_ids = np.asarray(sample_ids)

    if not is_train:
        return X, None, sample_ids

    if y_path is None:
        raise ValueError("训练数据必须提供标签文件 y_path")

    y_path = Path(y_path)

    if y_path.suffix.lower() in [".xlsx", ".xls"]:
        labels_df = pd.read_excel(y_path, header=None)
    else:
        labels_df = pd.read_csv(y_path, header=None)

    labels_df = labels_df.iloc[:, :2].copy()
    labels_df.columns = ["filename", "label"]

    def normalize_filename(value):
        value = str(value).strip()

        if not value.lower().endswith(".csv"):
            value += ".csv"

        return value

    labels_df["filename"] = labels_df["filename"].map(normalize_filename)
    labels_df["label"] = pd.to_numeric(
        labels_df["label"],
        errors="raise"
    )

    label_map = dict(
        zip(labels_df["filename"], labels_df["label"])
    )

    y = np.asarray(
        [label_map[sample_id] for sample_id in sample_ids],
        dtype=np.float32
    )

    return X, y, sample_ids

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / (len(y_true) - 1))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, r2, sep, mae

if __name__ == "__main__":
    base_path = Path(r"C:\Users\zmzhang\Desktop\NIR_model")

    X_train, y_train, sample_ids_train = load_data(
        base_path / "train",
        y_path=base_path / "y_train.xlsx",
        is_train=True
    )

    X_test, y_test, sample_ids_test = load_data(
        base_path / "test",
        is_train=False
    )

    processor = DataProcessor(X_train, y_train)

    preprocess_methods = ['derivative'] 
    feature_selection_methods = ['recursive_elimination']  

    count = 400  

    processed_X_train = X_train.copy()
    processed_X_test = X_test.copy()
    count = min(count, processed_X_train.shape[1])

    if 'airPLS' in preprocess_methods:
        print("应用airPLS基线校正...")
        processed_X_train = processor.baseline_correction_airPLS(processed_X_train, lambda_=100, itermax=15)
        processed_X_test = processor.baseline_correction_airPLS(processed_X_test, lambda_=100, itermax=15)

    if 'MSC' in preprocess_methods:
        print("应用多元散射校正...")
        processed_X_train = processor.perform_msc(processed_X_train)
        processed_X_test = processor.perform_msc(processed_X_test)

    if 'SNV' in preprocess_methods:
        print("应用标准正态变量变换...")
        processed_X_train = processor.perform_standard_normal_variate(processed_X_train)
        processed_X_test = processor.perform_standard_normal_variate(processed_X_test)

    if 'Savitzky-Golay' in preprocess_methods:
        print("应用Savitzky-Golay平滑滤波...")
        processed_X_train = processor.perform_savgol(processed_X_train)
        processed_X_test = processor.perform_savgol(processed_X_test)

    if 'detrend' in preprocess_methods:
        print("应用去趋势...")
        processed_X_train = processor.perform_detrend(processed_X_train)
        processed_X_test = processor.perform_detrend(processed_X_test)

    if 'derivative' in preprocess_methods:
        print("一阶微分")
        processed_X_train = processor.spectral_first_order_derivative(processed_X_train)
        processed_X_test = processor.spectral_first_order_derivative(processed_X_test)

    for method in feature_selection_methods:
        if method == 'uniform_sampling':
            print(f"应用均匀采样 (count={count})...")
            sampled_train, sampled_test = processor.uniform_sampling(
                processed_X_train, 
                processed_X_test, 
                count=count  
            )
            processed_X_train = sampled_train
            processed_X_test = sampled_test
        elif method == 'univariant_selection':
            print("应用单变量特征选择...")
            selected_indices = processor.perform_univariant_selection(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]
        elif method == 'recursive_elimination':
            print("应用递归特征消除...")
            selected_indices = processor.perform_recursive_elimination(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]
        elif method == 'pca':
            print("应用主成分分析...")
            pca_train = processor.perform_pca(processed_X_train, count=count)
            pca_test = processor.perform_pca(processed_X_test, count=count)
            processed_X_train = pca_train
            processed_X_test = pca_test
        elif method == 'uve':
            print("应用无信息变量消除...")
            selected_indices = processor.perform_uve(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]

    print("数据处理完成！")

    models = {
        'PLSR': PLSRegression(),
        'TabPFN': TabPFNRegressor(
            model_path=model_path,
            device=device,
            random_state=42
        )
    }

    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name == 'PLSR':
            # 使用网格搜索调整主成分数
            param_grid = {'n_components': [5, 10, 15, 20, 25, 30, 35, 40]}
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            grid_search.fit(processed_X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            best_model = model
            best_params = 'Default params'
            setattr(best_model, "ignore_pretraining_limits", True)
            best_model.fit(processed_X_train, y_train)
        
        if y_train is not None:
            y_pred_train = best_model.predict(processed_X_train)
            if hasattr(y_pred_train, "ravel"):
                y_pred_train = y_pred_train.ravel()
            rmse_train, r2_train, sep_train, mae_train = evaluate_model(y_train, y_pred_train)
            print(f"{model_name} - RMSE (Train): {rmse_train:.4f}, R² (Train): {r2_train:.4f}, SEP (Train): {sep_train:.4f}, MAE (Train): {mae_train:.4f}")

        y_pred_test = best_model.predict(processed_X_test)
        if hasattr(y_pred_test, "ravel"):
            y_pred_test = y_pred_test.ravel()
        results[model_name] = {'Predicted': y_pred_test}
        print(f"{model_name} - Predictions on test set: {y_pred_test[:5]}")  # 打印前5个预测值

    # 保存预测结果到Excel
    for model_name, model_results in results.items():
        predictions_df = pd.DataFrame({
            'Sample_ID': sample_ids_test,
            'Predicted': model_results['Predicted']
        })
        predictions_df.to_excel(f'predictions_{model_name}.xlsx', index=False)


    print("Predictions saved to 'predictions_{model_name}.xlsx' files.")

