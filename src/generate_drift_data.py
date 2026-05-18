import os
import pandas as pd
import numpy as np

def generate_drifted_credit_data(n_samples=1000, random_state=100):
    # Cố định seed nhưng dùng số khác (100) để phân phối khác đi
    np.random.seed(random_state)
    
    # 1. Thu nhập (~18 triệu, thấp hơn so với tập train)
    mu = np.log(18000000) 
    thu_nhap = np.random.lognormal(mean=mu, sigma=0.4, size=n_samples)
    thu_nhap = np.round(thu_nhap.clip(5000000, 100000000), -5)
    
    # 2. Thời hạn vay ngắn hơn (Chỉ còn 6, 12, 18, 24 tháng)
    thoi_han_vay = np.random.choice([6, 12, 18, 24], size=n_samples)
    
    # 3. Số tiền vay (Gấp 15 - 25 lần thu nhập tháng)
    he_so_vay = np.random.uniform(15, 25, n_samples)
    so_tien_vay = np.round((thu_nhap * he_so_vay), -6).clip(10000000, 3000000000)

    # 4. Điểm tín dụng (~420)
    diem_tin_dung = 420 + np.random.normal(0, 60, n_samples)
    diem_tin_dung = diem_tin_dung.clip(300, 850).astype(int)

    # 5. Tính lãi suất & Trả hàng tháng
    lai_suat_nam = 0.08 + (1 - (diem_tin_dung / 850)) * 0.15 
    lai_suat_thang = lai_suat_nam / 12

    tks = (1 + lai_suat_thang) ** thoi_han_vay
    tra_hang_thang = so_tien_vay * lai_suat_thang * tks / (tks - 1)

    # LOGIC NỢ XẤU: Dựa trên tỷ lệ nợ trên thu nhập hàng tháng (DTI)
    # Nếu tiền trả hàng tháng > 50% thu nhập -> Rất dễ nợ xấu
    ty_le_tra_no = tra_hang_thang / thu_nhap
    diem_rui_ro = (ty_le_tra_no * 4) - (diem_tin_dung / 600) + np.random.normal(0, 0.5, n_samples)

    xac_suat_no_xau = 1 / (1 + np.exp(-(diem_rui_ro - 1.5)))
    lich_su_no_xau = np.random.binomial(1, xac_suat_no_xau)

    # Gom dữ liệu vào DataFrame
    df = pd.DataFrame({
        'thu_nhap': thu_nhap.astype(int),
        'so_tien_vay': so_tien_vay.astype(int),
        'thoi_han_vay': thoi_han_vay,
        'diem_tin_dung': diem_tin_dung,
        'tra_hang_thang': tra_hang_thang.astype(int),
        'lich_su_no_xau': lich_su_no_xau
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    
    # Lấy toàn bộ data từ generator
    df_drift = generate_drifted_credit_data(n_samples=1000)
    
    # Lưu thành file riêng data/drift_data.csv
    df_drift.to_csv('data/drift_data.csv', index=False)
    
    print("--------------------------------------------------")
    print("THÀNH CÔNG: Đã tạo file dữ liệu Drift đầy đủ biến")
    print("--------------------------------------------------")
    print(f"Thu nhập TB thời kỳ drift: {df_drift['thu_nhap'].mean():,.0f} VND")
    print(f"Tiền trả hàng tháng TB: {df_drift['tra_hang_thang'].mean():,.0f} VND")
    print(f"Tỷ lệ nợ xấu thời kỳ drift: {df_drift['lich_su_no_xau'].mean()*100:.1f}%")