import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

history_files = glob.glob("*.csv")
if not history_files:
    raise FileNotFoundError("Không tìm thấy file history CSV nào trong thư mục hiện tại.")

os.makedirs("plots", exist_ok=True)

for file in history_files:
    model_name = os.path.splitext(os.path.basename(file))[0]
    print(f"Đang xử lý: {model_name}")

    df = pd.read_csv(file)

    # Vẽ loss cho gender và age
    plt.figure(figsize=(10, 5))
    plt.plot(df['gender_output_loss'], label='Gender Loss')
    plt.plot(df['val_gender_output_loss'], label='Val Gender Loss')
    plt.plot(df['age_output_loss'], label='Age Loss')
    plt.plot(df['val_age_output_loss'], label='Val Age Loss')
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{model_name}_loss.png")
    plt.close()

    # Vẽ accuracy cho gender
    if 'gender_output_accuracy' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['gender_output_accuracy'], label='Gender Accuracy')
        plt.plot(df['val_gender_output_accuracy'], label='Val Gender Accuracy')
        plt.title(f"{model_name} - Gender Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{model_name}_gender_accuracy.png")
        plt.close()

    # Vẽ MAE cho age
    if 'age_output_mae' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['age_output_mae'], label='Age MAE')
        plt.plot(df['val_age_output_mae'], label='Val Age MAE')
        plt.title(f"{model_name} - Age MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{model_name}_age_mae.png")
        plt.close()

print("✅ Đã tạo biểu đồ và lưu trong thư mục 'plots'.")
