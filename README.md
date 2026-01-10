# 🌞 Solar Power Generation Forecasting using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project focuses on predicting future solar power generation using **Artificial Neural Networks (ANN)** trained on historical solar and weather data. Accurate solar power forecasting helps energy planners anticipate supply variations and efficiently allocate electrical resources across cities, ensuring grid stability and sustainable energy management.

The system integrates data preprocessing, exploratory statistical analysis, ANN-based prediction, and visualization into a unified workflow.

---

## 🎯 Objectives
- Predict short-term solar power generation using historical data
- Analyze the influence of environmental parameters on power output
- Support smart energy allocation and urban resource planning
- Provide clear visual insights for decision-making

---

## 🧠 Technologies Used
- **Python**
- **Artificial Neural Networks (ANN)**
- **Pandas & NumPy** – Data processing
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn / TensorFlow / Keras** – Model training
- **Streamlit** – Interactive application interface

---

## 📂 Dataset Description
The project uses structured tabular datasets containing meteorological and solar generation data.

### Files Used
- `solarpowergeneration.csv` – Hourly solar and weather data
- `Solar per day con..xlsx` – Daily consolidated records

### Key Features
- Temperature (°C)
- Relative Humidity (%)
- Wind Speed (m/s)
- Solar Irradiance (W/m²)
- Cloud Cover
- Angle of Incidence
- Generated Power (kW)

### Sample (Shortened View)

| Temperature (°C) | Humidity (%) | Wind Speed (m/s) | Irradiance (W/m²) | Power (kW) |
|-----------------|--------------|------------------|------------------|------------|
| 2.17 | 31 | 6.37 | 0.00 | 454.10 |
| 3.65 | 33 | 4.68 | 108.58 | 2214.85 |

📎 **Dataset Source:**  
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

---

## 📊 Statistical Analysis
Before model training, statistical analysis is performed to understand data behavior:
- Mean, median, and standard deviation computation
- Missing and outlier value detection
- Feature correlation analysis

### Visualizations Used
- Histogram – Solar power distribution
- Bar chart – Monthly average power
- Scatter plot – Power vs Irradiance
- Boxplot – Hourly power variation
- Pie chart – Feature contribution

---

## ⚙️ Methodology
1. **Data Collection & Cleaning**
2. **Feature Selection & Normalization**
3. **Exploratory Data Analysis (EDA)**
4. **ANN Model Design & Training**
5. **Prediction & Performance Evaluation**
6. **Visualization using Streamlit**

---

## 📈 Expected Outcome
- Accurate short-term solar power forecasts
- Improved grid stability and load balancing
- Better allocation of energy resources for cities
- Support for renewable energy planning and sustainability

---

## 🚀 Future Scope
- Integration with real-time weather APIs
- City-level or region-based forecasting
- Hybrid models (ANN + LSTM)
- Battery storage optimization
- Deployment on cloud platforms

---

## 📚 References
- Abuella & Chowdhury, *Random Forest Ensemble of SVR Models for Solar Power Forecasting*, 2017  
- Kumari et al., *Deep Learning Models for Solar Irradiance Forecasting*, 2021  
- El Hendouzi, *Solar Photovoltaic Power Forecasting: A Review*, 2020  
- Rahimi et al., *Comprehensive Review on Ensemble Solar Forecasting*, 2023  
- Konstantinou et al., *Stacked LSTM for PV Forecasting*, 2021  

---

## 👨‍💻 Author
**Dhavala V D M Adithya Naidu**  
Computer Science & Engineering (AI & ML)   

---

## ⭐ Acknowledgement
This project was developed for academic and research purposes to explore the application of Artificial Neural Networks in renewable energy forecasting.

If you find this project useful, please ⭐ star the repository!
