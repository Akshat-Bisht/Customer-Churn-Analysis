# 📊 Customer Churn Analysis & Automated Risk Insights  
### Using Python, Pandas, NumPy, and Matplotlib

This project analyzes customer churn data to uncover insights, identify churn drivers, and generate automated risk tags that help businesses proactively address customer retention challenges.

---

## 🚀 Project Overview

Customer churn is one of the biggest problems for subscription-based and service-based businesses. This project performs:

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Churn trend identification  
- Rule-based churn risk prediction  
- Visualization of key insights  
- Automated churn-risk tagging  

The goal is to understand **why customers leave** and **which customer segments are at higher risk**, helping business teams take data-backed actions.

---

## 🧰 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| **Python** | Core programming language |
| **Pandas** | Data cleaning & preprocessing |
| **NumPy** | Numerical computations |
| **Matplotlib / Seaborn** | Data visualization |
| **Jupyter Notebook** | Analysis environment |

---

## 📁 Project Structure

Customer-Churn-Analysis/
│── data/
│ └── churn_data.csv
│── notebooks/
│ └── churn_analysis.ipynb
│── images/
│ ├── churn_distribution.png
│ ├── feature_correlations.png
│ └── risk_segments.png
│── src/
│ ├── preprocess.py
│ ├── visualize.py
│ └── risk_tagging.py
│── README.md
│── requirements.txt


---

## 🔍 Key Steps & Methodology

### **1️⃣ Data Cleaning**
- Handled missing values  
- Standardized column names  
- Encoded categorical variables  
- Managed outliers  
- Converted dates & numeric fields

### **2️⃣ Exploratory Data Analysis**
Analyzed:
- Churn distribution  
- Demographics  
- Subscription & usage behavior  
- Payment behavior  
- Correlations between variables

Visualizations include:
- Heatmaps  
- Count plots  
- Trend graphs  
- Boxplots  

### **3️⃣ Feature Engineering**
Created additional features such as:
- Tenure buckets  
- Contract type categories  
- Total service usage  
- Monthly charge patterns  
- Payment method indicators  

### **4️⃣ Rule-Based Churn Risk Tagging**
Based on EDA findings, customers are segmented into:
- **High Risk**
- **Medium Risk**
- **Low Risk**

Rules derived from data include:
- High monthly charges  
- Long gaps in activity  
- No contract / monthly contract  
- Multiple service drop-offs  
- Payment failures  

### **5️⃣ Insights Generated**
- Identified top churn-driving factors  
- Found high-risk demographic groups  
- Mapped behavior patterns linked to churn  
- Delivered visual & statistical summaries  

---

## 📈 Results & Findings

Some insights (dummy examples, replace with yours):

- Customers on **monthly contracts** have the highest churn rate.  
- Electronic check payment users show higher churn.  
- Longer tenure customers churn significantly less.  
- High monthly charges strongly correlate with churn.  

The automated risk module helps businesses target:
- Customers likely to churn soon  
- Customers with declining usage  
- Customers with billing complaints or payment failures  

---

## 🔮 Future Enhancements
- Add ML-based churn prediction (Logistic Regression / Random Forest)  
- Deploy REST API for business integration  
- Add dashboard using Streamlit or Flask  
- Cloud deployment on AWS/GCP  

---

## 📦 Installation

git clone https://github.com/Akshat-Bisht/Customer-Churn-Analysis
cd Customer-Churn-Analysis
pip install -r requirements.txt


---

## ▶️ Running the Project

Run Jupyter Notebook:

jupyter notebook notebooks/churn_analysis.ipynb


or execute the risk tagging script:

python src/risk_tagging.py


---

## 🤝 Contributing
Feel free to open issues or submit pull requests.

---

## 📬 Contact  
**Akshat Bisht**  
Email: akshatbisht7777@gmail.com  
GitHub: [Akshat-Bisht](https://github.com/Akshat-Bisht)
