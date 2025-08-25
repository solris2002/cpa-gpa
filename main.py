import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# =============== PAGE LAYOUT ===============
st.set_page_config(page_title="Dự đoán kết quả học tập", layout="wide")
st.markdown("""
    <style>
      .block-container { padding-left: 2rem; padding-right: 2rem; }
      .field-error { color: #d32f2f; font-size: 0.85rem; margin-top: 0.25rem; }
      .label-error { color: #d32f2f !important; font-weight: 600; }
      .caption-tight { margin-bottom: 0.25rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN")

# =============== HELPERS ===============
@st.cache_resource
def load_model(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model không tìm thấy: {path}")
    return joblib.load(path_obj)

def build_feature_vector(gpa_list, tc_list):
    features = []
    for gpa, tc in zip(gpa_list, tc_list):
        features.append(gpa)
        features.append(tc)
    return np.array(features).reshape(1, -1)

def parse_gpa(raw: str):
    try:
        if raw.strip() == "":
            return None, "Chưa nhập"
        v = float(raw)
        if not (0.0 <= v <= 4.0):
            return None, "GPA phải nằm trong khoảng 0.00 đến 4.00"
        return v, None
    except ValueError:
        return None, "GPA không phải là số hợp lệ"

def parse_tc(raw: str):
    try:
        if raw.strip() == "":
            return None, "Chưa nhập"
        v = int(raw)
        if v < 0:
            return None, "Tín chỉ phải là số nguyên không âm"
        return v, None
    except ValueError:
        return None, "Tín chỉ không phải là số nguyên hợp lệ"

# =============== SIDEBAR ===============
st.sidebar.subheader("Cài đặt đầu vào")

student_type = st.sidebar.selectbox("Định hướng sinh viên", ("Cử nhân", "Kỹ sư"))
max_semester = 6 if student_type == "Cử nhân" else 8
current_semester = st.sidebar.selectbox("Kỳ đã học", list(range(1, max_semester + 1)))

st.sidebar.divider()
st.sidebar.markdown(
    """
**Hướng dẫn nhanh**
- Chọn *Định hướng sinh viên* và *Kỳ đã học* ở trên.
- Nhập **GPA** và **Tín chỉ** cho từng kỳ đã học.
- Nhấn **Dự đoán** để xem *CPA tốt nghiệp* và *GPA kỳ kế tiếp* (nếu còn kỳ).

> Lưu ý: GPA trong khoảng **0.00–4.00**, Tín chỉ là **số nguyên không âm**.
    """
)

# =============== FORM (GRID, VALIDATE) ===============
# Dùng text_input để phân biệt 'chưa nhập' vs '0'
with st.form("input_form"):
    st.subheader("Nhập GPA và tín chỉ từng kỳ (bắt buộc)")

    n_cols = 4 if current_semester >= 6 else 3
    gpa_inputs_raw, tc_inputs_raw = [], []

    # Chúng ta sẽ validate sau khi bấm submit; để highlight lỗi từng ô,
    # mình sẽ render thông báo lỗi ngay dưới field (nếu có).
    for i in range(1, current_semester + 1):
        if (i - 1) % n_cols == 0:
            cols = st.columns(n_cols, gap="small")

        col = cols[(i - 1) % n_cols]
        with col:
            st.caption(f"Kỳ {i}", help=None)
            gpa_raw = st.text_input(
                f"GPA kỳ {i}",
                placeholder="VD: 3.25",
                key=f"gpa_raw_{i}",
                label_visibility="visible",
            )
            tc_raw = st.text_input(
                f"Tín chỉ kỳ {i}",
                placeholder="VD: 15",
                key=f"tc_raw_{i}",
                label_visibility="visible",
            )
        gpa_inputs_raw.append(gpa_raw)
        tc_inputs_raw.append(tc_raw)

    submitted = st.form_submit_button("Dự đoán")

# =============== VALIDATE ===============
if not submitted:
    st.info("Điền đầy đủ GPA và tín chỉ rồi nhấn 'Dự đoán' để xem kết quả.")
    st.stop()

gpa_vals, tc_vals, errors, per_field_errors = [], [], [], {}

for idx, (gpa_raw, tc_raw) in enumerate(zip(gpa_inputs_raw, tc_inputs_raw), start=1):
    gpa, err_gpa = parse_gpa(gpa_raw)
    tc,  err_tc  = parse_tc(tc_raw)

    if err_gpa:
        errors.append(f"- Kỳ {idx}: GPA: {err_gpa}")
        per_field_errors[f"gpa_raw_{idx}"] = err_gpa
    if err_tc:
        errors.append(f"- Kỳ {idx}: Tín chỉ: {err_tc}")
        per_field_errors[f"tc_raw_{idx}"] = err_tc

    gpa_vals.append(gpa)
    tc_vals.append(tc)

# Nếu có lỗi, hiển thị tổng hợp + gợi ý sửa
if errors:
    st.error("⚠️ Có lỗi với dữ liệu nhập:\n" + "\n".join(errors))
    st.stop()

# =============== RENDER HIGHLIGHT TẠI CHỖ (sau submit) ===============
# Streamlit không hỗ trợ thay đổi viền input per-widget một cách chính thống,
# nên mình hiển thị lỗi ngay dưới các field khi có lỗi (đã stop() nếu có lỗi).
# Nếu bạn muốn tô đỏ label ngay khi có lỗi mà vẫn tiếp tục hiển thị,
# bỏ st.stop() ở trên và thêm đoạn bên dưới để vẽ lỗi inline:

# for idx in range(1, current_semester + 1):
#     if f"gpa_raw_{idx}" in per_field_errors:
#         st.markdown(f"<div class='field-error'>Kỳ {idx} - GPA: {per_field_errors[f'gpa_raw_{idx}']}</div>", unsafe_allow_html=True)
#     if f"tc_raw_{idx}" in per_field_errors:
#         st.markdown(f"<div class='field-error'>Kỳ {idx} - Tín chỉ: {per_field_errors[f'tc_raw_{idx}']}</div>", unsafe_allow_html=True)

# =============== BUILD INPUT VECTOR ===============
input_data = build_feature_vector(gpa_vals, tc_vals)

# =============== PREDICT ===============
model_prefix = "8" if student_type == "Cử nhân" else "10"

# CPA tốt nghiệp
group_key_cpa = f"GPA_TC_1_{current_semester}" if current_semester > 1 else "GPA_TC_1"
cpa_model_path = f"models_streamlit/final_cpa_{model_prefix}_ki.joblib"

try:
    cpa_dict = load_model(cpa_model_path)
    if group_key_cpa not in cpa_dict:
        st.error(f"Không tìm thấy key '{group_key_cpa}' trong model CPA.")
        st.stop()
    model_cpa = cpa_dict[group_key_cpa]
    predicted_cpa = model_cpa.predict(input_data)[0]
    st.subheader("🎓 Dự đoán CPA tốt nghiệp")
    st.success(f"CPA Tốt Nghiệp: {predicted_cpa:.2f}")
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Đã xảy ra lỗi lúc dự đoán CPA: {e}"); st.stop()

# GPA kỳ tiếp theo (nếu có)
if current_semester < max_semester:
    group_key_gpa = f"GPA_{current_semester + 1}"
    next_gpa_path = f"models_streamlit/next_gpa_{model_prefix}_ki.joblib"
    try:
        next_dict = load_model(next_gpa_path)
        if group_key_gpa not in next_dict:
            st.error(f"Không tìm thấy key '{group_key_gpa}' trong model GPA kế tiếp.")
            st.stop()
        model_next = next_dict[group_key_gpa]
        predicted_next_gpa = model_next.predict(input_data)[0]
        st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}")
        st.info(f"GPA dự đoán: {predicted_next_gpa:.2f}")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Đã xảy ra lỗi lúc dự đoán GPA kỳ tiếp theo: {e}")

