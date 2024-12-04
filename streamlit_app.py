# Import thư viện
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# Thiết lập tiêu đề ứng dụng
st.set_page_config(page_title="Phân Tích Dữ Liệu Click Quảng Cáo", layout="wide")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .title {
        font-size: 50px; /* Tăng kích thước tiêu đề */
        color: #ff5733; /* Màu sắc nổi bật cho tiêu đề */
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s; /* Thêm hiệu ứng chuyển động */
    }
    .title:hover {
        transform: scale(1.1); /* Hiệu ứng phóng to khi hover */
    }
    .subtitle {
        font-size: 30px; /* Tăng kích thước tiêu đề phụ */
        color: #333;
        font-weight: bold;
        margin: 15px 0;
    }
    .description {
        font-size: 20px;
        color: #555;
        margin: 10px 0;
    }
    .search-box {
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<div class="title">Phân Tích Dữ Liệu Click Quảng Cáo</div>', unsafe_allow_html=True)



# Thanh tìm kiếm
search_term = st.text_input("Tìm kiếm thông tin trong dữ liệu:", "")


# Tải dữ liệu
url = "https://raw.githubusercontent.com/DS-PNQ/adlick/refs/heads/main/ad_click_dataset.csv"
data = pd.read_csv(url)

# Tìm kiếm trong dữ liệu nếu có từ khóa
if search_term:
    st.subheader("Kết quả tìm kiếm")
    filtered_data = data[data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    st.write(filtered_data)
else:
    # Hiển thị dữ liệu
    st.subheader("Dữ liệu ban đầu")
    st.write(data)

# Loại bỏ các cột không cần thiết
data = data.drop(columns=['id', 'full_name'], axis=1)

# Thông tin dữ liệu
st.write(f"Kích thước dữ liệu: {data.shape}")
st.write(data.info())

# Phân loại các cột số và cột phân loại
num_cols = data.select_dtypes(include=['float64', 'int64'])
cat_cols = data.select_dtypes(include=['object'])

# Hiển thị biến số
st.subheader('Biến số:')
st.write(num_cols.columns.tolist())

# Hiển thị biến phân loại
st.subheader("Biến phân loại:")
st.write(cat_cols.columns.tolist())

# Mô tả dữ liệu
st.subheader("Mô tả dữ liệu")
st.write(data.describe().T)

# Thống kê giá trị 0, null và giá trị duy nhất
for feature in num_cols:
    zero_values = (data[feature] == 0).sum()
    null_values = data[feature].isnull().sum()
    unique_values = len(data[feature].unique())

    st.write(f"**Biến:** {feature}")
    st.write(f"Số lượng giá trị 0: {zero_values}")
    st.write(f"Số lượng giá trị null: {null_values}")
    st.write(f"Số lượng giá trị duy nhất: {unique_values}")
    st.write("=" * 30)

# Kiểm tra giá trị null
st.subheader("Kiểm tra giá trị null")
st.write(data.isnull().sum())

# Biểu đồ nhiệt cho giá trị null
st.subheader("Biểu đồ nhiệt cho các giá trị null")
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_title('Biểu đồ nhiệt cho các giá trị null trong từng cột')
sns.heatmap(data.isnull(), ax=ax, cmap='viridis')
st.pyplot(fig)

# Thay thế các giá trị null
data['gender'] = data['gender'].fillna('Unknown')
data['device_type'] = data['device_type'].fillna('Unknown')
data['ad_position'] = data['ad_position'].fillna('Unknown')
data['browsing_history'] = data['browsing_history'].fillna('Unknown')
data['time_of_day'] = data['time_of_day'].fillna('Unknown')

# Biểu đồ phân phối tuổi
st.subheader("Phân phối tuổi")
fig, ax = plt.subplots()
ax.hist(data['age'], bins=20, edgecolor='black')
ax.set_title('Phân phối tuổi')
ax.set_xlabel('Tuổi')
ax.set_ylabel('Tần suất')
st.pyplot(fig)

# Hàm KNN Imputer
def knn_impute(data, n_neighbors=5):
    data_encoded = data.copy()
    category_mappings = {}
    for col in data_encoded.select_dtypes(include='object').columns:
        data_encoded[col] = data_encoded[col].astype('category').cat.codes
        category_mappings[col] = dict(enumerate(data[col].astype('category').cat.categories))

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = pd.DataFrame(knn_imputer.fit_transform(data_encoded), columns=data_encoded.columns)

    for col in data.select_dtypes(include='object').columns:
        data_imputed[col] = data_imputed[col].round().astype(int).map(category_mappings[col])

    return data_imputed

data_imputed = knn_impute(data, n_neighbors=5)
data = data_imputed

# Kiểm tra lại giá trị null
st.subheader("Giá trị null sau khi impute")
st.write(data.isnull().sum())

# Biểu đồ phân phối tuổi sau khi impute
st.subheader("Phân phối tuổi sau khi impute")
fig, ax = plt.subplots()
ax.hist(data['age'], bins=20, edgecolor='black')
ax.set_title('Phân phối tuổi')
ax.set_xlabel('Tuổi')
ax.set_ylabel('Tần suất')
st.pyplot(fig)

# Tính toán ma trận tương quan
corr = num_cols.corr()
top_corr = corr['click'].sort_values(ascending=False)[1:20].to_frame()
st.subheader("Ma trận tương quan")
st.write(top_corr)

# Biểu đồ nhiệt cho ma trận tương quan
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', fmt=".2f")
ax.set_title('Ma trận tương quan')
st.pyplot(fig)

# Mã hóa các biến phân loại
data_encoded = pd.get_dummies(data, drop_first=True)
corr_matrix = data_encoded.corr()

# Biểu đồ nhiệt cho ma trận tương quan bao gồm các biến phân loại
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', linewidths=0.5)
ax.set_title('Ma trận tương quan bao gồm các biến phân loại')
st.pyplot(fig)

# Tính toán phân phối click
distribution_click = data['click'].value_counts(normalize=True) * 100
st.subheader("Phân phối click")
st.write(distribution_click)

# Phân phối theo nhóm tuổi
bins = [17, 24, 34, 44, 54, 64, 100]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
grouped = (
    data.assign(age_group=pd.cut(data['age'], bins=bins, labels=labels))
    .groupby(['age_group', 'click'], observed=False)
    .size()
    .unstack(fill_value=0)
)

percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
grouped.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Trạng thái Click theo nhóm tuổi')
ax.set_xlabel('Nhóm tuổi')
ax.set_ylabel('Số lượng Click')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(['Không Click', 'Click'])

# Thêm nhãn phần trăm lên cột
for i in range(len(grouped)):
    total = grouped.iloc[i].sum()
    for j in range(len(grouped.columns)):
        if total > 0:
            percentage = (grouped.iloc[i, j] / total) * 100
            ax.text(i, grouped.iloc[i, :j + 1].sum() - grouped.iloc[i, j] / 2, f'{percentage:.1f}%',
                    ha='center', va='center', color='white', fontsize=8)

st.pyplot(fig)

# Thống kê các loại thiết bị
st.subheader("Tỷ lệ click theo loại thiết bị")
click_counts = data[data['click'] == 1]['device_type'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(10, 6))
click_counts.plot(kind='bar', color='darkblue', ax=ax)
ax.set_xlabel('Loại thiết bị')
ax.set_ylabel('Tỷ lệ click')
ax.set_title('Tỷ lệ click theo loại thiết bị')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

st.pyplot(fig)

# Đếm số lần xuất hiện của từng loại thiết bị
st.subheader("Số lần xuất hiện của từng loại thiết bị")
st.write(data['device_type'].value_counts())

# Thống kê về ad_position
st.subheader("Tỷ lệ click theo vị trí quảng cáo")
click_counts = data[data['click'] == 1]['ad_position'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(10, 6))
click_counts.plot(kind='bar', color='darkblue', ax=ax)
ax.set_xlabel('Vị trí quảng cáo')
ax.set_ylabel('Tỷ lệ click')
ax.set_title('Tỷ lệ click theo vị trí quảng cáo')

for i in range(len(click_counts)):
    percentage = click_counts.iloc[i]
    ax.text(i, percentage / 2, f'{percentage:.1f}%',
            ha='center', va='center', color='white', fontsize=10)

st.pyplot(fig)

# Biểu đồ tương quan giữa ad_position và device_type
st.subheader("Số lượng Vị trí quảng cáo theo Loại thiết bị")
crosstab = pd.crosstab(data['device_type'], data['ad_position'])
fig, ax = plt.subplots(figsize=(10, 6))
crosstab.plot(kind='bar', color=sns.color_palette("pastel"), ax=ax)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Loại thiết bị', fontsize=12)
ax.set_ylabel('Số lượng', fontsize=12)
ax.set_title('Số lượng Vị trí quảng cáo theo Loại thiết bị', fontsize=14)
ax.legend(title='Vị trí quảng cáo', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
st.pyplot(fig)

# Biểu đồ độ tuổi theo loại thiết bị
st.subheader("Độ tuổi theo loại thiết bị")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='device_type', y='age', hue='device_type', data=data, palette='Set3', ax=ax)
ax.set_title('Độ tuổi theo loại thiết bị')
st.pyplot(fig)

# Biểu đồ độ tuổi theo thời gian trong ngày
st.subheader("Độ tuổi theo thời gian trong ngày")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='time_of_day', y='age', hue='time_of_day', data=data, palette='Set3', ax=ax)
ax.set_title('Độ tuổi theo thời gian trong ngày')
plt.xticks(rotation=45)
st.pyplot(fig)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------
# 1. Streamlit App Configuration
# ------------------------------
st.set_page_config(
    page_title="Logistic Regression Evaluation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Logistic Regression Model Evaluation")

# ------------------------------
# 2. Load Data
# ------------------------------

@st.cache(allow_output_mutation=True)
def load_data():
   
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=10, 
                               n_informative=5, n_redundant=2, 
                               n_classes=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(1, 11)]
    data = pd.DataFrame(X, columns=feature_names)
    data['click'] = y
    return data

data = load_data()

st.subheader("Dataset Overview")
st.write("### Features:", data.drop('click', axis=1).columns.tolist())
st.write("### Target:", 'click')
st.dataframe(data.head())

# ------------------------------
# 3. Data Preparation
# ------------------------------

st.subheader("Data Preparation")

# Convert categorical features to numeric (if any). In this sample, data is already numeric.
# If your dataset has categorical features, uncomment and modify the following lines:
# X = data.drop('click', axis=1)
# y = data['click']
# X = pd.get_dummies(X, drop_first=True)

X = data.drop('click', axis=1)
y = data['click']

# Display class distribution
st.markdown("**Class Distribution:**")
class_dist = y.value_counts().reset_index()
class_dist.columns = ['Class', 'Count']
st.bar_chart(class_dist.set_index('Class'))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.markdown("**Training and Testing Split:**")
col1, col2 = st.columns(2)
with col1:
    st.write("### Training Set")
    st.write(X_train.shape)
with col2:
    st.write("### Testing Set")
    st.write(X_test.shape)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.markdown("**Feature Scaling Applied:** StandardScaler")

# ------------------------------
# 4. Train Logistic Regression Model
# ------------------------------

st.subheader("Logistic Regression Model Training")

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

st.success("Logistic Regression model trained successfully!")

# ------------------------------
# 5. Make Predictions
# ------------------------------

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------
# 6. Evaluate the Model
# ------------------------------

st.subheader("Model Evaluation")

# 6.1 Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Accuracy", value=f"{accuracy:.2%}")

# 6.2 Classification Report
st.markdown("**Classification Report:**")
classification_rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()
st.dataframe(classification_df.style.format("{:.2f}"))

# 6.3 Confusion Matrix
st.markdown("**Confusion Matrix:**")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# 6.4 ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

st.markdown("**ROC Curve:**")
fig_roc, ax_roc = plt.subplots()
sns.lineplot(x=fpr, y=tpr, label=f'AUC = {roc_auc:.2f}')
sns.lineplot([0, 1], [0, 1], linestyle='--', color='gray')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
ax_roc.legend(loc='lower right')
st.pyplot(fig_roc)

st.markdown(f"**AUC:** {roc_auc:.2f}")

# ------------------------------
# 7. Coefficients Table and Visualization
# ------------------------------

st.subheader("Model Coefficients")

# Create a DataFrame for coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Calculate the absolute value of coefficients for sorting
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()

# Sort the coefficients by absolute value in descending order
coefficients_sorted = coefficients.sort_values(by='Abs_Coefficient', ascending=False)

# Display the coefficients table
st.markdown("**Logistic Regression Coefficients:**")
st.dataframe(coefficients_sorted.drop('Abs_Coefficient', axis=1).reset_index(drop=True))

# Plot coefficients
st.markdown("**Coefficient Bar Chart:**")
fig_coeff, ax_coeff = plt.subplots(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_sorted, palette='viridis', ax=ax_coeff)
ax_coeff.set_title('Logistic Regression Coefficients')
ax_coeff.set_xlabel('Coefficient Value')
ax_coeff.set_ylabel('Feature')
plt.tight_layout()
st.pyplot(fig_coeff)

# ------------------------------
# 8. Optional: Download Results
# ------------------------------

st.subheader("Download Results")

# Convert classification report to CSV
st.markdown("**Download Classification Report:**")
csv_classification = classification_df.to_csv(index=True)
st.download_button(
    label="Download Classification Report",
    data=csv_classification,
    file_name='classification_report.csv',
    mime='text/csv',
)

# Convert coefficients to CSV
st.markdown("**Download Coefficients:**")
csv_coefficients = coefficients_sorted.drop('Abs_Coefficient', axis=1).reset_index(drop=True).to_csv(index=False)
st.download_button(
    label="Download Coefficients Table",
    data=csv_coefficients,
    file_name='coefficients_table.csv',
    mime='text/csv',
)
)
st.pyplot(fig_roc)
