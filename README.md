# 🏡 EstateIQ — Gurgaon Real Estate Price Prediction

An end-to-end Machine Learning project that predicts real estate property prices in Gurgaon using property-related features. Built with Python, Scikit-learn, and Streamlit.

<p align="center">
  <a href="https://estateiq-h4cbhu7hyc7p9ljbfhcffv.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

## 🚀 Live Demo

**Try the application:**  
https://estateiq-h4cbhu7hyc7p9ljbfhcffv.streamlit.app/

## ✨ Features

- Predict property prices using a trained regression model.
- Interactive Streamlit interface.
- Property selection and filtering based on available options.
- Data processing and machine learning model integration.
- Online deployment through Streamlit Community Cloud.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Regression
- **Visualization:** Plotly
- **Web Application:** Streamlit
- **Model Storage:** Joblib
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
EstateIQ/
├── app.py
├── requirements.txt
├── README.md
├── property_filters.py
├── data/
│   └── data of gurugram real Estate.csv
├── models/
│   ├── gurgaon_price_model_validated.pkl
│   └── model_metrics_validated.json
└── tests/
    └── test_property_filters.py
```

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/shaswat44/EstateIQ.git
   cd EstateIQ
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

## 🎯 Project Objective

The objective of EstateIQ is to demonstrate how machine learning regression can be integrated into an interactive web application to estimate real estate prices and make property data more accessible.

## 👨‍💻 Author

**Shaswat Bhatnagar**

- GitHub: [shaswat44](https://github.com/shaswat44)
