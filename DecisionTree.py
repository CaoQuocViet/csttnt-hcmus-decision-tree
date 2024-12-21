# Lab04: Decision Tree

# MSSV: 22810218
# Họ và Tên: Cao Quốc Việt

### Cách làm bài tập ###
# SV sẽ làm bài trực tiếp trên notebook này; bạn cần **hoàn thành những phần code đánh dấu với từ `TODO`**.
# SV cần ghi thêm giải thích cho phần code của mình (Có thể dưới dạng comment hoặc text block riêng).
#
# SV có thể trao đổi ý tưởng với các bạn cùng lớp cũng như tìm kiếm thông tin từ internet, sách, v.v...; nhưng *bài tập về nhà này phải là của bạn.*
#
### Cách nộp bài tập ###
# Trước khi nộp bài, hãy chạy lại notebook (`Kernel` ->`Restart & Run All`).
#
# Sau đó đổi tên notebook thành `MSSV` (ví dụ: nếu MSSV của bạn là 1234567 thì đặt tên notebook là `1234567.ipynb`) và nộp lên moodle.
#
### Tham khảo: ###
# 1. https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f
# 2. https://www.kaggle.com/code/jebathuraiibarnabas/decision-tree-id3-from-scratch


### Import library
# Nhập các thư viện cần thiết cho việc phân tích dữ liệu và xây dựng mô hình cây quyết định
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

### Load Iris dataset
# Tải tập dữ liệu Iris, một trong những bộ dữ liệu phổ biến cho việc thử nghiệm với các thuật toán học máy
iris = datasets.load_iris()
X = iris.data  # Dữ liệu đầu vào
y = iris.target  # Nhãn của dữ liệu

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## 1. Decision Tree: Iterative Dichotomiser 3 (ID3)

### 1.1 Information Gain

def entropy(counts, n_samples):
    """
    Tính entropy cho một tập dữ liệu
    Parameters:
    -----------
    counts: danh sách số lượng mẫu trong mỗi lớp
    n_samples: tổng số mẫu trong tập dữ liệu
    -----------
    return: giá trị entropy
    """
    probabilities = [count / n_samples for count in counts]  # Tính xác suất của mỗi lớp
    entropy_value = -sum(p * np.log2(p) for p in probabilities if p > 0)  # Tính entropy, tránh log(0)
    return entropy_value

def entropy_of_one_division(division):
    """
    Tính entropy cho một nhóm dữ liệu đã được chia
    """
    n_samples = len(division)  # Số lượng mẫu trong phân chia
    n_classes = set(division)  # Các lớp trong phân chia

    counts = []
    # Đếm số mẫu trong mỗi lớp và lưu vào danh sách counts
    for cls in n_classes:
        counts.append(np.sum(division == cls))

    return entropy(counts, n_samples), n_samples  # Trả về entropy và số mẫu

def get_entropy(y_predict, y):
    """
    Tính entropy cho một phân chia dựa trên quyết định ngưỡng
    """
    n = len(y)
    entropy_true, n_true = entropy_of_one_division(y[y_predict])  # Entropy bên trái
    entropy_false, n_false = entropy_of_one_division(y[~y_predict])  # Entropy bên phải
    
    # Entropy tổng thể
    s = (n_true / n) * entropy_true + (n_false / n) * entropy_false
    return s

# 1.2 Decision Tree

class DecisionTreeClassifier:
    def __init__(self, tree=None, depth=0):
        '''Khởi tạo cây quyết định
        Parameters:
        -----------------
        tree: cây quyết định
        depth: độ sâu của cây quyết định sau khi huấn luyện'''

        self.depth = depth  # Lưu độ sâu của cây
        self.tree = tree  # Cây quyết định

    def fit(self, X, y, node={}, depth=0):
        '''Huấn luyện mô hình cây quyết định
        Parameters:
        -----------------
        X: dữ liệu huấn luyện
        y: nhãn của dữ liệu huấn luyện
        ------------------
        return: node của cây quyết định
        '''
        
        # Điều kiện dừng

        # Nếu tất cả giá trị của y đều giống nhau
        if np.all(y == y[0]):
            return {'val': y[0]}  # Trả về giá trị của lớp
        else:
            # Tìm phân chia tốt nhất dựa trên thông tin gain
            col_idx, cutoff, entropy = self.find_best_split_of_all(X, y)  
            y_left = y[X[:, col_idx] < cutoff]  # Nhãn cho phần bên trái
            y_right = y[X[:, col_idx] >= cutoff]  # Nhãn cho phần bên phải
            node = {'index_col': col_idx,
                    'cutoff': cutoff,
                    'val': np.mean(y)}  # Giá trị trung bình của nhãn
            node['left'] = self.fit(X[X[:, col_idx] < cutoff], y_left, {}, depth + 1)  # Đệ quy cho bên trái
            node['right'] = self.fit(X[X[:, col_idx] >= cutoff], y_right, {}, depth + 1)  # Đệ quy cho bên phải
            self.depth += 1  # Tăng độ sâu
            self.tree = node  # Cập nhật cây
            return node

    def find_best_split_of_all(self, X, y):
        # Tìm phân chia tốt nhất trên toàn bộ dữ liệu
        col_idx = None
        min_entropy = float('inf')  # Bắt đầu với vô cùng
        cutoff = None
        for i, col_data in enumerate(X.T):  # Duyệt qua từng cột dữ liệu
            entropy, cur_cutoff = self.find_best_split(col_data, y)  # Tìm phân chia tốt nhất cho cột
            if entropy == 0:  # Nếu entropy bằng 0 thì đã tìm thấy phân chia hoàn hảo
                return i, cur_cutoff, entropy
            elif entropy < min_entropy:  # Cập nhật nếu tìm thấy entropy nhỏ hơn
                min_entropy = entropy
                col_idx = i
                cutoff = cur_cutoff

        return col_idx, cutoff, min_entropy  # Trả về chỉ số cột, ngưỡng và entropy nhỏ nhất

    def find_best_split(self, col_data, y):
        ''' Tìm phân chia tốt nhất cho một cột dữ liệu
        Parameters:
        -------------
        col_data: dữ liệu của một cột'''

        min_entropy = float('inf')  # Bắt đầu với vô cùng
        cutoff = None

        # Duyệt qua col_data để tìm ngưỡng với entropy nhỏ nhất
        for value in set(col_data):
            y_predict = col_data < value  # Dự đoán nhãn dựa trên ngưỡng
            my_entropy = get_entropy(y_predict, y)  # Tính entropy cho phân chia này
            # Cập nhật nếu tìm thấy entropy nhỏ hơn
            if my_entropy < min_entropy:
                min_entropy = my_entropy
                cutoff = value

        return min_entropy, cutoff  # Trả về entropy nhỏ nhất và ngưỡng tương ứng

    def predict(self, X):
        # Dự đoán nhãn cho dữ liệu mới
        tree = self.tree
        pred = np.zeros(shape=len(X))  # Khởi tạo mảng dự đoán
        for i, c in enumerate(X):
            pred[i] = self._predict(c)  # Dự đoán cho từng hàng
        return pred

    def _predict(self, row):
        # Dự đoán cho một hàng dữ liệu cụ thể
        cur_layer = self.tree
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']  # Đi xuống bên trái
            else:
                cur_layer = cur_layer['right']  # Đi xuống bên phải
        else:
            return cur_layer.get('val')  # Trả về giá trị cuối cùng

### 1.3 Phân loại trên tập dữ liệu Iris

# Tạo mô hình cây quyết định và huấn luyện với dữ liệu huấn luyện
model = DecisionTreeClassifier()
tree = model.fit(X_train, y_train)

# Dự đoán trên tập huấn luyện và tính độ chính xác
pred = model.predict(X_train)
print('Accuracy of your decision tree model on training data:', accuracy_score(y_train, pred))

# Dự đoán trên tập kiểm tra và tính độ chính xác
pred = model.predict(X_test)
print('Accuracy of your decision tree model:', accuracy_score(y_test, pred))
