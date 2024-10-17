import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Tải bộ dữ liệu chữ số
digits = datasets.load_digits()

# Trực quan hóa một vài chữ số đầu tiên
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, label in zip(axes.flatten(), digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Dự đoán: {label}')
plt.show()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

# Tạo bộ phân loại KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán dữ liệu kiểm tra
y_pred = knn.predict(X_test)

# Tính toán ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)

print("Ma trận nhầm lẫn:")
print(conf_matrix)
print(f"Độ chính xác: {accuracy:.2f}")

# Vẽ ma trận nhầm lẫn với các giá trị số trong ô
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap=plt.cm.Blues, fignum=1)
plt.title('Ma trận nhầm lẫn')
plt.colorbar()
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')

# Thêm số vào từng ô trong ma trận nhầm lẫn
for (i, j), value in np.ndenumerate(conf_matrix):
    plt.text(j, i, f'{value}', ha='center', va='center', color='red')

plt.show()

# ---------------------- Giao diện tkinter ----------------------

# Khởi tạo giao diện tkinter
root = tk.Tk()
root.title("Nhận dạng chữ số bằng KNN")

# Canvas để vẽ chữ số
canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky=tk.W)

# Biến
image = Image.new("L", (200, 200), (255))
draw = ImageDraw.Draw(image)

# Hàm để vẽ trên canvas
def paint(event):
    x1, y1 = event.x - 8, event.y - 8
    x2, y2 = event.x + 8, event.y + 8  # Sửa lỗi ở đây
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)

canvas.bind("<B1-Motion>", paint)

# Hàm để xóa canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 200, 200], fill="white")

# Hàm để dự đoán chữ số
def predict_digit():
    # Thay đổi kích thước và xử lý trước hình ảnh
    img = image.resize((8, 8))
    img_data = list(img.getdata())
    img_data = [255 - x for x in img_data]
    img_data = [x / 16 for x in img_data]

    # Dự đoán chữ số
    prediction = knn.predict([img_data])[0]
    result_var.set(f"Kết quả là: {prediction}")

# Các nút chức năng
result_var = tk.StringVar()
tk.Button(root, text="Chạy", command=predict_digit).grid(row=0, column=1, padx=5)
tk.Button(root, text="Xóa", command=clear_canvas).grid(row=0, column=2, padx=5)
tk.Label(root, textvariable=result_var).grid(row=1, column=0, columnspan=3)

# Chạy ứng dụng
root.mainloop()
