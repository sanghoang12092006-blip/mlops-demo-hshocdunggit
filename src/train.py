import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Đảm bảo các thư mục tồn tại
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Generating dummy data...")
    # i. Dùng make_classification sinh dữ liệu giả lập
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        random_state=42
    )

    # Transform X to make it look like realistic data (e.g. positive values in millions)
    # X features are normally distributed around 0. We'll shift and scale them.
    import numpy as np
    thu_nhap_raw = np.abs(X[:, 0]) * 20 + 10 
    so_tien_vay_raw = np.abs(X[:, 1]) * 100 + 10

    # Tạo DataFrame với 3 cột và định dạng kiểu dữ liệu rõ ràng
    df = pd.DataFrame({
        'thu_nhap': thu_nhap_raw.astype(int),       # Kiểu số nguyên (triệu VNĐ)
        'so_tien_vay': so_tien_vay_raw.astype(int), # Kiểu số nguyên (triệu VNĐ)
        'lich_su_no_xau': y.astype(int)             # Kiểu số nguyên (0 hoặc 1)
    })

    # ii. Lưu dữ liệu vào data/train_data.csv
    data_path = 'data/train_data.csv'
    df.to_csv(data_path, index=False)
    print(f"Data saved to: {data_path}")

    # iii. Huấn luyện mô hình ML
    print("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(df[['thu_nhap', 'so_tien_vay']], df['lich_su_no_xau'])

    # iv. Lưu mô hình vào models/model.pkl
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
     
if __name__ == "__main__":
    main()
