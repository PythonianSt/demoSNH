import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

st.set_page_config(page_title="BMI Prediction App", layout="wide")

# Load the height and weight dataset for linear regression
htwt_data = pd.read_csv('htwt.csv')

# Convert heights from inches to cm and create model
htwt_data['Height(cm)'] = htwt_data['Height(Inches)'] * 2.54
X = htwt_data[['Height(Inches)']]
y = htwt_data['Weight(Pounds)']

# Train the linear regression model for weight prediction
model = LinearRegression()
model.fit(X, y)

# Load BMI data for clustering and classification
bmi_data = pd.read_csv('bmi.csv')

# Clean BMI data: remove non-numeric values
bmi_data['BMI'] = pd.to_numeric(bmi_data['BMI'], errors='coerce')
bmi_data.dropna(subset=['BMI'], inplace=True)  # Drop rows with NaN BMI

# Prepare clustering with KMeans
bmi_values = bmi_data['BMI'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(bmi_values)
bmi_data['Cluster'] = kmeans.labels_

# Train Random Forest classifier for predicting BMI result category (0-4)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(bmi_data[['Sex', 'Age', 'Weight', 'Height', 'BMI']], bmi_data['Result'])

# Train Decision Tree for visualization purposes
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_classifier.fit(bmi_data[['Sex', 'Age', 'Weight', 'Height', 'BMI']], bmi_data['Result'])

# Set up the title
st.title("DHV Machine Learning (ML) Predictions")
st.write("Demo Linear Regression, Clustering, K-Means, and Random Forest Classification")

# User inputs
sex = st.selectbox("Select Sex:", ["Male", "Female"])
age = st.number_input("Enter your age:", min_value=0)
sex_numeric = 1 if sex == "Female" else 2

# User input for height in cm
height_cm = st.number_input("Enter your height in centimeters:", min_value=0.0)

# Convert height to inches for weight prediction
height_inches = height_cm / 2.54

# Predict weight in pounds and kg based on height
predicted_weight_pounds = model.predict([[height_inches]])[0]
predicted_weight_kg = predicted_weight_pounds * 0.453592

# User input for actual weight in kg
actual_weight_kg = st.number_input("Enter your actual weight in kilograms:", min_value=0.0)

# Calculate BMIs for both predicted and actual weights
if height_cm > 0:
    predicted_bmi = predicted_weight_kg / ((height_cm / 100) ** 2)
    actual_bmi = actual_weight_kg / ((height_cm / 100) ** 2)
    
    # Predict Result category using Random Forest classifier based on predicted BMI
    predicted_result = rf_classifier.predict([[sex_numeric, age, predicted_weight_kg, height_cm, predicted_bmi]])[0]
    
    # Predict Result category based on actual BMI
    actual_result = rf_classifier.predict([[sex_numeric, age, actual_weight_kg, height_cm, actual_bmi]])[0]
    
    # Determine if actual BMI is an inlier or outlier based on clustering
    actual_cluster = kmeans.predict([[actual_bmi]])[0]
    is_inlier = actual_cluster == kmeans.predict([[predicted_bmi]])[0]
else:
    predicted_bmi = actual_bmi = np.nan
    predicted_result = actual_result = None
    is_inlier = None

# Display results
st.write(f"### Predicted Weight:")
st.write(f"- **Weight (Kilograms):** {predicted_weight_kg:.2f} kg")
st.write(f"### Calculated BMIs:")
if predicted_result is not None:
    st.write(f"- **Predicted BMI:** {predicted_bmi:.2f}")
    #st.write(f"- **Predicted Result Category:** {predicted_result} (0=สมส่วน, 1=ท้วม, 2=อ้วน, 3=ผอม, 4=ผอมมาก)")
    st.write(f"- **Actual BMI:** {actual_bmi:.2f}")
    #st.write(f"- **Actual Result Category:** {actual_result} (0=สมส่วน, 1=ท้วม, 2=อ้วน, 3=ผอม, 4=ผอมมาก)")
    st.markdown(f"<h2 style='color:blue;'><strong>Predicted Result Category:</strong> {actual_result}(0=สมส่วน, 1=ท้วม, 2=อ้วน, 3=ผอม, 4=ผอมมาก)", unsafe_allow_html=True)
    #st.write(f"- **Outlier/Inlier Status:** {'Inlier' if is_inlier else 'Outlier'}")
else:
    st.write("- **BMI:** Invalid height input, please enter a height greater than 0 cm.")

# Plotting the BMI data, predicted BMI, and actual BMI
plt.figure(figsize=(10, 5))
plt.scatter(bmi_data['BMI'], [0] * len(bmi_data), color='black', label='BMI Data')
if predicted_result is not None:
    plt.scatter(predicted_bmi, 0, color='green', s=100, edgecolor='black', label='Predicted BMI')
    plt.scatter(actual_bmi, 0, color='blue' if is_inlier else 'red', s=100, edgecolor='black', label='Actual BMI')
plt.title('BMI Clustering (N=5,332)')
plt.xlabel('BMI Values')
plt.yticks([])
plt.legend(loc='upper right')
st.pyplot(plt)


# Display Decision Tree diagram for the Result prediction
st.subheader("Decision Tree for Result Prediction: ข้อมูลดัชนีมวลกาย (BMI) ของประชาชนไทยตามเกณฑ์กรมพลศึกษา")
dot_data = export_graphviz(
    dt_classifier, 
    out_file=None, 
    feature_names=['Sex', 'Age', 'Weight', 'Height', 'BMI'], 
    class_names=['0', '1', '2', '3', '4'],
    filled=True, 
    rounded=True, 
    special_characters=True
)
st.graphviz_chart(dot_data)

