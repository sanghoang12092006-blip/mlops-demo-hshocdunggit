from fastapi import FastAPI     # framework tạo API
from pydantic import BaseModel  # Kiểm tra dữ liệu đầu vào
import joblib                   # Thư viện để load model đã được huấn luyện sẵn
import pandas as pd             # Thư viện xử lý dữ liệu dạng bảng
import csv                      # Thư viện để ghi log vào file CSV
import os                       # Thư viện để kiểm tra sự tồn tại của file
from datetime import datetime   # Thư viện để lấy thời gian hiện tại

# Khởi tạo app
app = FastAPI()

# Load model từ Dev 1 tạo ra
model = joblib.load("models/model.pkl")

# Định nghĩa dữ liệu đầu vào
class InputData(BaseModel):
    thu_nhap: float
    so_tien_vay: float
    lich_su_no_xau: int

# Endpoint dự đoán
@app.post("/predict")
def predict(data: InputData):
    # Dự đoán
    df = pd.DataFrame([data.model_dump()])
    ket_qua = model.predict(df)[0]

    # Ghi log
    log_path = "logs/inference_logs.csv"
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["thoi_gian", "thu_nhap", "so_tien_vay", "lich_su_no_xau", "ket_qua"])
        writer.writerow([datetime.now(), data.thu_nhap, data.so_tien_vay, data.lich_su_no_xau, ket_qua])

    return {"ket_qua_du_doan": int(ket_qua)}