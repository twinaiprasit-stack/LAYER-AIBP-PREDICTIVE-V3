# 🥚 Layer-X Hybrid Dashboard (CPF Edition) — UX 2025

> **AI-Powered Market Intelligence Dashboard**  
> พัฒนาเพื่อการพยากรณ์ราคาไข่ไก่ ด้วยโมเดล Prophet + XGBoost  
> พร้อมระบบ Hybrid Ensemble และการออกแบบ UX สไตล์ Layer-X  

---

## ⚙️ Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Frontend / App** | Streamlit |
| **Forecasting** | Prophet  |
| **Machine Learning** | XGBoost , Scikit-learn  |
| **Visualization** | Plotly , Matplotlib  |
| **PDF Export & Font** | ReportLab , Kaleido  |

---

## 🧠 Key Features

### 🥇 Page 1 — Egg Price Forecast Dashboard
- แสดงผลการ **พยากรณ์ 8 สัปดาห์ข้างหน้า** ด้วย Prophet Model  
- มี **Confidence Interval Band** และ Interactive Tooltip  
- แสดง **KPI Cards:** Min / Avg / Max Forecast (฿ ต่อ ฟอง)  
- ใช้ดีไซน์ **Layer-X Glass Glow Theme** พร้อม **Egg Rocket Animation**  
- รองรับ **Export PDF Report ภาษาไทย** พร้อมโลโก้ CPF และกราฟ  

### 🚀 Page 2 — Hybrid Model Performance (Prophet + XGBoost)
- เปรียบเทียบ **Actual vs Prophet vs XGBoost vs Hybrid**  
- Hybrid Model = การเฉลี่ยผลระหว่าง Prophet และ XGBoost  
- แสดง **Model KPI:** MAE, RMSE, MAPE, Accuracy (จาก Offline Evaluation)  
- รองรับ **Export PDF Summary** พร้อมกราฟและตรา CPF  
- ออกแบบให้ขยายต่อได้สำหรับ **Dynamic KPI (เฉพาะช่วงกราฟ)**  

---

## 🧩 Design & UX

- 🎨 **Layer-X Gradient Background:** Deep Blue → Cyan  
- 🚀 **Egg Rocket Floating Animation** มุมขวาบน  
- 💡 **Glass Card KPI Style** พร้อม Glow Effect  
- 🌙 **Dark Mode Chart Theme** (Plotly)  
- 🧾 **PDF Export Theme** สอดคล้องกับ Corporate CPF CI  

---

## 📁 Project Structure
```
layer-aibp-predictive-v3/
│
├── app_layerx_hybrid_v4_2.py        # Main Streamlit App (2 Pages)
│
├── prophet_model.pkl                # Trained Prophet Model
├── xgboost_model.pkl                # Trained XGBoost Model
├── scaler_and_features.pkl          # Scaler & Feature Metadata
├── prophet_forecast.csv             # Prophet Forecast Output
├── Predict Egg Price ... Test.csv   # Actual Market Prices (Test Data)
│
├── assets/
│   ├── LOGO-CPF.jpg                 # CPF Corporate Logo
│   ├── egg_rocket.png               # Rocket Icon (Animated)
│   ├── space_bg.png                 # Background Image
│
└── requirements.txt                 # Environment Dependencies
```

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app_layerx_hybrid_v4_2.py
```

> 💡 **Tips:**  
> - หากรันบน Streamlit Cloud ให้แน่ใจว่ามีไฟล์ model และ assets ครบ  
> - ตรวจสอบการติดตั้ง `kaleido` เพื่อให้การ Export PDF ทำงานถูกต้อง  

---

## 📊 Model Interpretation

- **Prophet**: จับแนวโน้มและฤดูกาลได้ดี (trend + seasonality)  
- **XGBoost**: เสริมความละเอียดจากปัจจัยภายนอก (feed price, supply index)  
- **Hybrid Model**: เพิ่มเสถียรภาพและความแม่นยำโดยเฉลี่ยผล  
- **KPI บนหน้า Dashboard** มาจากผล **Offline Evaluation Set** ซึ่งสะท้อนความแม่นยำในภาพรวม  
  > (ในบางช่วง Prophet อาจเกาะ Actual มากกว่า เนื่องจากเป็นช่วงข้อมูลสั้นเฉพาะกราฟ 8 สัปดาห์)

---

## 🧾 PDF Export
- รองรับภาษาไทยเต็มรูปแบบ (ฟอนต์ `HYSMyeongJo-Medium`)  
- แสดง Logo CPF + Egg Rocket + Graph + Timestamp + KPI  
- สามารถปรับโทนสี, เพิ่ม Watermark หรือ Footer ได้ตาม CPF CI  

---

## 🧑‍💼 Use Case Example
> ใช้ในการประชุม **War Room / Executive Review**  
> เพื่อสรุปแนวโน้มราคาไข่ไก่รายสัปดาห์  
> เหมาะสำหรับการแสดงผลบน **Layer-X War Room (6-screen setup)**  
> หรือส่งเป็นรายงาน PDF สำหรับผู้บริหารระดับ VP / BU Head  

---

## 🪄 Future Roadmap

- 🔹 เพิ่ม **Dynamic KPI Calculation** (คำนวณ Accuracy เฉพาะช่วงที่เลือกบนกราฟ)  
- 🔹 เพิ่ม **Model Comparison Mode** (Prophet v1 vs v2 vs Hybrid)  
- 🔹 เพิ่ม **AI Voice Summary** สำหรับสรุปแนวโน้มอัตโนมัติ  
- 🔹 รองรับ **Multi-Business Dashboard** (Feed, Layer, Grading)  
- 🔹 ปรับปรุง UI ให้เหมาะกับ **War Room Display Mode**

---

### 👨‍🚀 Maintainer
**Layer-X Digital Transformation Team (Layer Business Unit, CPF)**  
> *Smart Data → Smart Decision → Smart Business*
