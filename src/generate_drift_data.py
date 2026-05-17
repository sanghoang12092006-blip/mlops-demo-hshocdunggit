import os
import pandas as pd
import numpy as np

def generate_drifted_credit_data(n_samples=1000, random_state=100):
    # Cố định seed nhưng dùng số khác (100) để phân phối khác đi
    np.random.seed(random_state)
    
    # KỊCH BẢN DRIFT: Khủng hoảng kinh tế toàn cầu
    # 1. Thu nhập mặt bằng chung giảm mạnh (Lúc train trung bình ~35tr, giờ còn ~18tr)
    mu = np.log(18000000) 
    thu_nhap = np.random.lognormal(mean=mu, sigma=0.4, size=n_samples)
    thu_nhap = np.round(thu_nhap.clip(5000000, 100000000), -5)
    
    # 2. Thời hạn vay ngắn hơn vì ngân hàng siết thắt chặt tín dụng
    thoi_han_vay = np.random.choice([6, 12, 18, 24], size=n_samples)
    
    # 3. Người dân túng quẫn nên có xu hướng vay liều (Gấp 15 đến 25 lần thu nhập)
    he_so_vay = np.random.uniform(15, 25, n_samples)
    so_tien_vay = np.round((thu_nhap * he_so_vay), -6).clip(10000000, 3000000000)

    # 4. Điểm tín dụng của người dân bị tụt giảm nghiêm trọng (Do nợ xấu lịch sử tăng)
    diem_tin_dung = 420 + np.random.normal(0, 60, n_samples)
    diem_tin_dung = diem_tin_dung.clip(300, 850).astype(int)

    # Gom dữ liệu vào DataFrame
    df = pd.DataFrame({
        'thu_nhap': thu_nhap.astype(int),
        'so_tien_vay': so_tien_vay.astype(int),
        'thoi_han_vay': thoi_han_vay,
        'diem_tin_dung': diem_tin_dung
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    # Sinh 1000 mẫu dữ liệu bị lệch pha (Drift)
    df_drift = generate_drifted_credit_data(n_samples=1000)
    
    # Lưu thành file riêng data/drift_data.csv
    df_drift.to_csv('data/drift_data.csv', index=False)
    
    print("--------------------------------------------------")
    print("THÀNH CÔNG: Đã tạo file dữ liệu Drift tại: data/drift_data.csv")
    print("--------------------------------------------------")
    print(f"Thu nhập TB thời kỳ drift: {df_drift['thu_nhap'].mean():,.0f} VND")
    print(f"Điểm tín dụng TB thời kỳ drift: {df_drift['diem_tin_dung'].mean():.1f} điểm")