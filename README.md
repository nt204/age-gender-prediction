# 🎯 Age & Gender Prediction with CNN (ResNet50, VGG16, EfficientNetB0)

## 📌 Giới thiệu
Dự án này sử dụng **mô hình CNN với 3 backbone khác nhau**:
- ResNet50
- VGG16
- EfficientNetB0  

Mục tiêu: Dự đoán **giới tính (classification)** và **tuổi (regression)** từ ảnh khuôn mặt người, dựa trên dataset **UTKFace**.

---

## 🗂 Dataset
- **Nguồn:** [UTKFace](https://susanqq.github.io/UTKFace/)
- **Kích thước ảnh:** 224x224 -> 128x128
- **Nhãn:** 
  - `Gender`: 0 = Nam, 1 = Nữ
  - `Age`: số nguyên

---

## 🏗 Cấu trúc model
- **2 outputs**:
  - `gender_output` → Sigmoid (Binary Crossentropy)
  - `age_output` → Linear (MAE)
- Loss tổng hợp = `loss_gender + loss_age`

---

## 📊 Kết quả huấn luyện
Biểu đồ dưới đây được tạo bằng script [`visualize.py`](/src/visualize.py).

### 🔹 Loss
| Model | Loss |
|-------|------|
| ResNet50 | ![](plots/ResNet50_loss.png) |
| VGG16 | ![](plots/VGG16_loss.png) |
| EfficientNetB0 | ![](plots/EfficientNetB0_loss.png) |

### 🔹 Gender Accuracy
| Model | Accuracy |
|-------|----------|
| ResNet50 | ![](plots/ResNet50_gender_accuracy.png) |
| VGG16 | ![](plots/VGG16_gender_accuracy.png) |
| EfficientNetB0 | ![](plots/EfficientNetB0_gender_accuracy.png) |

### 🔹 Age MAE
| Model | MAE |
|-------|-----|
| ResNet50 | ![](plots/ResNet50_age_mae.png) |
| VGG16 | ![](plots/VGG16_age_mae.png) |
| EfficientNetB0 | ![](plots/EfficientNetB0_age_mae.png) |

---

## 🚀 Cách chạy lại
jupyter notebook notebooks/EDA_and_Demo.ipynb
python src/train.py
