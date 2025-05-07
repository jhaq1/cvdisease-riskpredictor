# <img src="1fac0.svg" alt="heart" width="40" style="vertical-align: middle;"> Cardiovascular Disease Risk Predictor Webapp

https://private-user-images.githubusercontent.com/196167950/441401606-b1bf070f-a005-41a6-94f0-daece786f809.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDY2NDQ1MzcsIm5iZiI6MTc0NjY0NDIzNywicGF0aCI6Ii8xOTYxNjc5NTAvNDQxNDAxNjA2LWIxYmYwNzBmLWEwMDUtNDFhNi05NGYwLWRhZWNlNzg2ZjgwOS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUwN1QxODU3MTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wMmI4MmI3YjdiZTMzYWFjZWZiMDNmNDMwMWZjZjhiYTllMjM4ZjM2OGQzMDBkM2NhYTBlNjE4YjU0YjFlODM2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.7qbHdnDDYtHF9eHjXE0-9se0qZySHWpwfh7_dyxKSU8

## Introduction

This project provides a powerful tool for predicting the likelihood of cardiovascular disease (CVD) based on user-input health indicators. Built using [Streamlit](https://streamlit.io/), the app allows users to provide health indicators that are common risk factors for CVD. Although for education purposes this is tool, is a proof-of-concept for applying machine learning to preventative health screenings. **The model is not meant to provide a diagnosis. Please consult your healthcare providor for personalized medical advice**.

🔗 [Public GitHub Repo](https://github.com/jhaq1/cvdisease-riskpredictor)
🌐 [Try the Streamlit Webapp](https://cvdisease-riskpredictor.streamlit.app/)

## 👤 Author

Dataset identified by: **Junaid H.**  
Code, model development, and deployment by: **Junaid H.**

## 📚 Background & Motivation

Cardiovascular disease (CVD) remains the **leading cause of death in the United States**, responsible for approximately **1 in every 5 deaths**—around **695,000 Americans in 2021 alone** (Centers for Disease Control and Prevention [CDC], 2023). Beyond its mortality impact, CVD is widespread: roughly **48% of all adults in the U.S.** have some form of cardiovascular condition, including coronary heart disease, heart failure, or hypertension (Virani et al., 2021). Despite its prevalence, CVD often goes undiagnosed until a major event, such as a heart attack or stroke, making **early detection and lifestyle intervention** critical for prevention and management.

Many CVD risk factors—such as high blood pressure, elevated cholesterol, obesity, and sedentary behavior—are measurable and modifiable. The motivation behind this project is to **empower individuals and healthcare learners to understand how these factors contribute to disease risk** and to offer a tool for interactive exploration of those relationships.

This web app provides an accessible way to input various clinical and behavioral variables (like systolic pressure, weight, cholesterol, smoking status, and more) and receive a real-time prediction from a neural network trained on patient data. Users can observe how changes in one or more features—such as reducing BMI or improving cholesterol—affect the predicted CVD risk. This kind of interactivity not only **supports better health literacy**, but also emphasizes the importance of modifiable risk factors in long-term cardiovascular health.

By simulating the impact of incremental changes to health metrics, the app can help users appreciate the preventive power of lifestyle modifications and routine screenings. In practice, early identification of individuals at high risk enables earlier treatment with medication, diet, exercise, and monitoring, which can dramatically reduce the likelihood of life-threatening complications and improve long-term outcomes (Arnett et al., 2019).

While not a clinical diagnostic tool, this application **illustrates how machine learning can be applied to real-world health scenarios**, especially in raising awareness and personalizing disease risk insights in a user-friendly interface.

## 📊 Data Source

The data used is from the **Cardiovascular Disease dataset** available via [Kaggle](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset). It includes anonymized records of over 70,000 patients with features:

- Age: measured in days as an int
- Gender: gender of the patient as a string (male/female)
- Systolic and Diastolic Blood Pressure: measured in mmHg as an int
- Total Cholesterol levels: measured on a 0-5 unit scale with each unit denoting a step-change of 20mg/dL as an int
- Total Glucose levels: measured on a 0-16 unit scale with each unit denoting a step-change of 1 mmol/L as an int
- Weight: measured in kg as an int
- Height: measured in centimeters as an int
- Smoking, Alcohol, and Physical Activity: stored as a boolean 0 or 1

During preprocessing the below changes were made to the input data:
- Age was modified from days to years
- Total Cholesterol was marked as categorical and booleanified with normal and above normal as 0 and well above normal as 1
- Total glucose was marked as categorical and booleanified with normal and above normal as 0 and wel above normal as 1

The target variable indicates the presence or absence of diagnosed cardiovascular disease.

## 🛠️ Key Packages

- `Streamlit` — for the web app frontend
- `TensorFlow/Keras` — for model development and training
- `Pandas` — for data manipulation
- `Scikit-learn` — for preprocessing and evaluation
- `Matplotlib/Seaborn` — for data visualization

## 🧠 Model Architecture

First the data was imported and preprocessed as described in the Data Source section. The neural network is a binary classifier built using `tf.keras.Sequential`. The architecture includes:

- Input Layer: matching the shape of the 11 input features
- Hidden Layers: 2 dense layers with ReLU activations. 1st layer is 16 features and 2nd is 8 features
- Output Layer: 1 unit with a sigmoid activation for binary classification

Custom metrics like **sensitivity** and **specificity** are implemented using Keras backend operations. The model is ran for 36 epochs. Initially, a dropout layer was included after each hidden layer but this was not making a meaningful impact to the model metrics and thus was removed to improve computational efficiency.
 
## 📈 Model Interpretability 

Four plots are included in the webapp to aid interpretability. First, a neaural network schematic is provided showing the input, hidden, and output layers to better understand the achitecture of the model. Second, a feature correlation plot is provided to aid users in understanding which features are most strongly or negatively correlated to the final prediction. This helps better understand the model and also aids users in understanding the health risks each feature has on CVD. Third, subplots of accuracy and loss are provided by epoch to instruct users on how well the model was trained. Similarly a sensitivity and specificity by epoch plot is provided in place of the accuracy and loss plot to understand the occurance of false positives and negatives. Lastly an ROC AUC plot is provided to enable interpretability. These help users gain insights into the model's behavior and confidence.
 
## ⚠️ Limitations

While this web application provides an interactive and educational tool for exploring cardiovascular disease (CVD) risk, the dataset’s limitations also present opportunities for future enhancements. A key constraint of the current model is the use of categorical rather than continuous measurements for critical biomarkers such as cholesterol and glucose. For example, instead of labeling cholesterol as "normal" or "high," more informative and clinically meaningful predictions could be made using actual lipid panel values, particularly the breakdown between high-density lipoprotein (HDL) and low-density lipoprotein (LDL). These specific components have been shown to play distinct roles in CVD risk (Goff et al., 2014). Similarly, substituting a single glucose measurement with hemoglobin A1C—a more stable marker of long-term blood sugar control—would improve risk prediction, especially for individuals with diabetes or prediabetes.

The absence of comorbid condition data is another notable limitation. Chronic diseases such as diabetes, chronic obstructive pulmonary disease (COPD), and mental health disorders (including anxiety and depression) are well-documented contributors to cardiovascular complications (Collins et al., 2020). Including these variables could significantly improve model performance and allow for a more nuanced understanding of how overlapping health burdens influence CVD outcomes.

Emerging research also emphasizes the connection between cardiovascular and oral health. Periodontal disease has been associated with increased systemic inflammation, which contributes to the development and progression of atherosclerosis and other cardiovascular conditions (Lockhart et al., 2012). Including dental health data—particularly the presence of periodontal disease—could add a valuable dimension to predictive models.

Lastly, the dataset lacks sociodemographic granularity. Information such as racial and ethnic identity, geographic region, and socioeconomic status would provide critical context. Studies have shown that healthcare disparities, including unequal access to preventative care and implicit provider bias, can significantly influence cardiovascular outcomes (Carnethon et al., 2017). By incorporating these factors, future iterations of this application could not only improve accuracy but also foster discussions around health equity.

## 📚 References

Arnett, D. K., Blumenthal, R. S., Albert, M. A., Buroker, A. B., Goldberger, Z. D., Hahn, E. J., ... & Ziaeian, B. (2019). 2019 ACC/AHA guideline on the primary prevention of cardiovascular disease. Journal of the American College of Cardiology, 74(10), e177–e232. https://doi.org/10.1016/j.jacc.2019.03.010

Centers for Disease Control and Prevention. (2023). Heart disease facts. https://www.cdc.gov/heartdisease/facts.htm

Virani, S. S., Alonso, A., Aparicio, H. J., Benjamin, E. J., Bittencourt, M. S., Callaway, C. W., ... & Tsao, C. W. (2021). Heart disease and stroke statistics—2021 update: A report from the American Heart Association. Circulation, 143(8), e254–e743. https://doi.org/10.1161/CIR.0000000000000950

Sulianova. (n.d.). Cardiovascular disease dataset [Dataset]. Kaggle. https://www.kaggle.com/sulianova/cardiovascular-disease-dataset

Scott-Mu. (n.d.). Capstone project repository [Source code]. GitHub. https://github.com/ds4ph-bme/capstone-project-scott-mu

Psavarmattas. (n.d.). Cardiovascular heart disease prediction model [Source code]. GitHub. https://github.com/psavarmattas/Cardiovascular-Heart-Disease-Prediction-Model/blob/master/main.ipynb

Cassnutt. (n.d.). Predicting heart disease [Source code]. GitHub. https://github.com/cassnutt/Predicting_heart_disease

Carnethon, M. R., Pu, J., Howard, G., Albert, M. A., Anderson, C. A. M., Bertoni, A. G., ... & American Heart Association Council on Epidemiology and Prevention. (2017). Cardiovascular health in African Americans: A scientific statement from the American Heart Association. Circulation, 136(21), e393–e423. https://doi.org/10.1161/CIR.0000000000000534

Collins, R., Reith, C., Emberson, J., Armitage, J., & Baigent, C. (2020). Interpretation of the evidence for the efficacy and safety of statin therapy. The Lancet, 388(10059), 2532–2561. https://doi.org/10.1016/S0140-6736(16)31357-5

Goff, D. C., Lloyd-Jones, D. M., Bennett, G., Coady, S., D’Agostino, R. B., Gibbons, R., ... & Wilson, P. W. (2014). 2013 ACC/AHA guideline on the assessment of cardiovascular risk: A report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines. Circulation, 129(25_suppl_2), S49–S73. https://doi.org/10.1161/01.cir.0000437741.48606.98

Lockhart, P. B., Bolger, A. F., Papapanou, P. N., Osinbowale, O., Trevisan, M., Levison, M. E., ... & Baddour, L. M. (2012). Periodontal disease and atherosclerotic vascular disease: Does the evidence support an independent association? Circulation, 125(20), 2520–2544. https://doi.org/10.1161/CIR.0b013e31825719f3


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bAw0TZc0)
