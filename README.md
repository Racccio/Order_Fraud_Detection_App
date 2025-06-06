
# 🚨 Fraud Detection in Last-Mile Delivery

A machine learning project to detect fraudulent behavior in last-mile delivery operations — such as MDND (Marked Delivered Not Delivered) — using order and rider data. Built using Python, Pandas, Scikit-learn, and Streamlit.

---

## 📦 Project Summary

In quick-commerce companies like **Blinkit**, fraudulent activities like marking an order as delivered without actually delivering it lead to financial losses and poor customer experiences. This project aims to automatically flag such suspicious delivery records using both unsupervised and supervised machine learning techniques.

---

## 📁 Dataset Structure

The project uses a synthetic delivery dataset with the following fields:

| Column Name         | Description                                      |
|---------------------|--------------------------------------------------|
| `order_id`          | Unique order identifier                          |
| `rider_id`          | Delivery partner ID                              |
| `pickup_time`       | Timestamp when the order was picked              |
| `delivery_time`     | Timestamp when the order was marked delivered    |
| `dropzone_time`     | Time spent in dropzone before handover           |
| `distance_km`       | Distance between store and customer              |
| `complaint_type`    | Complaint category (e.g., MDND, missing, none)   |
| `is_fraud`          | Label (1 = fraud, 0 = normal)                    |

---

## 🧠 Features Created

- `delivery_duration_min`: Time between pickup and delivery
- `delivery_speed_kmph`: Calculated speed = distance/time
- `dropzone_lag_min`: Time from pickup to dropzone scan

---

## 🤖 Models Used

### 1. **Isolation Forest**
- Unsupervised anomaly detection
- Flags unusually fast deliveries and speed outliers

### 2. **Logistic Regression**
- Supervised classification using known fraud labels
- Predicts whether a delivery is likely fraudulent

---

## 📊 Streamlit App

A web dashboard built with **Streamlit** allows:
- Uploading a delivery CSV
- Real-time fraud prediction
- Interactive visualization (scatter plots, fraud count, etc.)

To run the app:

```bash
streamlit run app.py
```

---

## 🛠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

---

## 💡 Future Improvements

- Integrate GPS coordinates for geospatial fraud analysis
- Add feedback loop for retraining based on confirmed frauds
- Support multi-city fraud pattern detection

---

## 👨‍💻 Author

**Rachit Kaushik**  
BCA – Data Science, Bennett University  
🔗 [LinkedIn](https://www.linkedin.com/in/rachitkaus)  
📧 kaushikrachit.3105@gmail.com

---

## 📄 License

MIT License – feel free to use and modify for educational or research purposes.
