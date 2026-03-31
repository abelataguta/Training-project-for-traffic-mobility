# Training-project-for-traffic-mobility

### 🚦 Smart Traffic AI Suite

A comprehensive Smart Traffic Intelligence and Prediction System built with Python, Streamlit, and Machine Learning, combining real-time traffic analytics and AI-powered traffic prediction for smart city applications.

#### 📌 Overview

    This project consists of two interactive dashboards:
    
    Traffic Intelligence Dashboard – Provides visual analytics, KPI metrics, maps, and trends for urban traffic data.
    Traffic Condition Prediction Dashboard – Uses a trained Random Forest model to predict traffic levels (Low, Medium, High) based on real-time inputs or batch datasets.
    
    Together, these apps provide actionable insights for traffic management, urban planning, and smart mobility systems.

#### 🎯 Features

**1️⃣ Traffic Intelligence Dashboard**

    KPIs: Total vehicles, average energy consumption, emission levels, total accidents
    Interactive Map: Visualize accidents and traffic locations on a Leaflet map
    Accident Analysis: Charts by month, year, and hour
    Numerical Analysis: Trends for energy consumption, traffic speed distribution
    Categorical Visualization: Traffic conditions vs weather, traffic light states
    Data Explorer: Expandable raw dataset viewer

**2️⃣ Traffic Condition Prediction Dashboard**

    Real-Time Traffic Prediction: Predict Low / Medium / High traffic conditions
    Prediction Confidence: Shows probability of each class
    Feature Importance: Interactive chart to understand model decision-making
    Batch Prediction: Upload CSV files to get bulk predictions and download results
    Confusion Matrix Demo: Optional model performance visualization

**🧱 Tech Stack**

    Python Libraries:
    Streamlit, Pandas, NumPy
    Matplotlib, Seaborn, Plotly
    Folium & streamlit_folium
    Scikit-learn (Random Forest)
    Joblib (model serialization)
    Tools: VS Code, Git, GitHub
    Deployment: Can be deployed to Streamlit Cloud or local server

#### 📂 Project Structure

    📁 smart-traffic-ai-suite/
    │
    ├── dashboard_intelligence/
    │   ├── app.py                 # Smart Traffic Intelligence Dashboard
    │   ├── smart_mobility_dataset.csv
    │   └── requirements.txt
    │
    ├── dashboard_prediction/
    │   ├── app.py                 # Traffic Condition Prediction Dashboard
    │   ├── rf_model.pkl
    │   └── requirements.txt
    │__ ppt/
    |   |__ DS_T00.pdf
    |   |__ DS_T01.pdf
    |   |__ DS_T02.pdf             # Document for reading 
    |   |__ DS_T03.pdf
    |   |__ DS_T04.pdf
    |   |__ DS_T05.pdf
    ├── README.md                  # This file

#### ⚙️ Installation & Setup

**1️⃣ Clone the Repository**

    git clone https://github.com/abelataguta/smart-traffic-ai-suite.git
        cd smart-traffic-ai-suite

**2️⃣ Create Virtual Environment**

    python -m venv venv
    
#### Activate Windows
    venv\Scripts\activate
#### Activate Mac/Linux
    source venv/bin/activate

**3️⃣ Install Dependencies**

    pip install -r dashboard_intelligence/requirements.txt
    pip install -r dashboard_prediction/requirements.txt

**4️⃣ Run Dashboards**

    Traffic Intelligence Dashboard:
    cd dashboard_intelligence
    streamlit run app.py

    Traffic Prediction Dashboard:
    cd dashboard_prediction
    streamlit run app.py

**🔮 Model Details (Prediction Dashboard)**

    ###### Algorithm: Random Forest Classifier
    
    **Input Features:**
    
    Latitude, Longitude, Vehicle Count, Traffic Speed, Road Occupancy, Traffic Light State, Weather Condition, Accident Report, Sentiment Score, Ride Sharing Demand, Parking Availability, Emission Levels, Energy Consumption, Hour, Day
    Output Classes:
    0 → Low
    1 → Medium
    2 → High

**📊 Visualizations**

    Interactive maps and charts (Intelligence Dashboard)
    Feature importance & probability distribution (Prediction Dashboard)
    Confusion matrix demo for evaluation
    
**🛠 Workflow for Contributions**

    Fork the repository
    Create a branch:
    git checkout -b feature-new-visualization
    
    Commit your changes:
    git add .
    git commit -m "Add new traffic visualization"
    
    Push branch:
    git push origin feature-new-visualization
    
    Create Pull Request on GitHub
    
**🚀 Future Enhancements**

    Add real-time traffic API integration
    Deploy dashboards on Streamlit Cloud or Heroku
    Integrate deep learning models for traffic forecasting (LSTM)
    Add explainable AI visualizations (SHAP, LIME)
    
**🧹 .gitignore Suggestion**

     env/
    **__pycache__/**
    *.py
    *.pkl
    .DS_Store
    .env

## 👨‍💻 Author

**Ababa Lata – Data Scientist | ML Engineer | Smart Mobility Researcher**
