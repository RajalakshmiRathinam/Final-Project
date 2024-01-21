from datetime import date
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.express as px

icon = Image.open("C:\Final Project\store.jpg")

# SETTING PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Store_Weekly_Sales_Prediction",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded")


st.title(':violet[Store_Weekly_Sales_Prediction]') 

with st.sidebar:
    selected = option_menu("Menu", ["Home","Explore","Insights","Top Chart","Weekly_Chart"],
                           icons =["house","image","toggles", "bar-chart-line","graph-up-arrow"],
                          default_index=0,
                          orientation="vertical",
                          styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "#FF0000"},
                                   "icon": {"font-size": "40px"},
                                   "container" : {"max-width": "2000px"},
                                   "nav-link-selected": {"background-color": "#D3D3D3"},
                                   "nav": {"background-color": "#D3D3D3"}})
    
# READING THE CLEANED DATAFRAME
df = pd.read_csv('C:\\Final Project\\finalsales.csv')

# HOME MENU
if selected == "Home":
    
    st.markdown(":black_large_square: **Project Title** : Store_Weekly_Sales_Prediction")

    technologies = "streamlit, Machine Learning"
    st.markdown(f":black_large_square: **Technologies** : {technologies}")

    overview = "Streamlit application that allows users are opening a new Store at a particular location. Now, Given the Store Location, Area, Size and other params. Predict the overall weekly sales of the Store."
    st.markdown(f":black_large_square: **Overview** : {overview}")
    st.image(Image.open("C:\Final Project\sales.jpeg"),width = 400)

# EXPLORE MENU
if selected == "Explore":      

    with st.form("my_form"):
        col1, col2, col3 = st.columns([0.5,0.5,0.1])
    
        with col1:
            Store = st.text_input(label='**Store(Min:1 & Max:45)**')
            Department = st.text_input(label='**Department(Min:1 & Max:99)**')  
            IsHoliday = st.text_input(label='**IsHoliday(Min:0 & Max:1)**')  
            Temperature = st.text_input(label='**Temperature(Min:-5.00 & Max: 105.00)**')
            CPI = st.text_input(label='**CPI(Min:100.0000 & Max: 250.0000)**')
            Unemployment = st.text_input(label='**Unemployment(Min:1.000 & Max: 20.000)**')
            Type = st.text_input(label='**Type(Min:1 & Max:3)**')

        with col2:    
            Size = st.text_input("**Size (Min:1 & Max:300000)**")
            Day = st.text_input(label='**Day(Min:1 & Max:31)**')
            Month = st.text_input(label='**Month(Min:1 & Max:12)**')
            Year = st.text_input(label='**Year(Min:2010 & Max:2012)**')
            Fuel_Price = st.text_input(label='**Fuel_Price(Min:1.000 & Max:5.000)**')
            Total_MarkDown = st.text_input(label='**Total_MarkDown(Min:0.00 & Max:170000.00)**')
            Expected_Weekly_Sales = st.text_input(label='**Expected_Weekly_Sales(Min:0.1 & Max:1000000)**')
        
        with col3:
            store = int(Store) if Store else None
            department = int(Department) if Department else None
            isholiday = int(IsHoliday) if IsHoliday else None
            temperature = float(Temperature) if Temperature else None
            cpi = float(CPI) if CPI else None
            unemp = float(Unemployment) if Unemployment else None
            type = int(Type) if Type else None
            size = float(Size) if Size else None  
            day = int(Day) if Day else None  
            month = int(Month) if Month else None   
            year = int(Year) if Year else None 
            fuel_price = np.log(float(Fuel_Price)) if Fuel_Price else None
            total_MarkDown = float(Total_MarkDown) if Total_MarkDown else None
            expected_weekly_sale = np.log(float(Expected_Weekly_Sales)) if Expected_Weekly_Sales else None
        
        
    # Form submission button
        submit_button = st.form_submit_button("Submit")

    # Load the model and scaler
        with open('C:\Final Project\dtmodel.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('C:\Final Project\scaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

            # Function to make predictions
            def predict_sales(sample):
                sample_scaled = scaler_loaded.transform(sample.reshape(1, -1))
                prediction = loaded_model.predict(sample_scaled)
                return np.exp(prediction)[0]
    
    # Use the user inputs for prediction
    if submit_button:
        # Create feature array
        Inputs = [store,department, isholiday,temperature,CPI,Unemployment,type, size,day,month,year,fuel_price,total_MarkDown,expected_weekly_sale]

         # Make prediction
        prediction_result = predict_sales(np.array(Inputs))
        st.success(f'Predicted Weekly Sales: ${prediction_result:.2f}')
        st.balloons()


st.set_option('deprecation.showPyplotGlobalUse', False)
if selected == "Insights": 
    
    
    tab1,tab2 = st.tabs(["$\huge Store $", "$\huge Department $"])
    
    with tab1:

        st.write("Top 10 Stores:")
        top_stores = df.groupby('Store')['Weekly_Sales'].sum().nlargest(10)
        st.bar_chart(top_stores) 
    with tab2:
        
        department_weekly_sales = df.groupby('Dept')['Weekly_Sales'].sum()
        top_departments = department_weekly_sales.nlargest(10)

        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        top_departments.plot(kind='bar', color='pink')
        plt.title('Top 10 Departments based on Sales')
        plt.xlabel('Department')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45, ha='right')
        st.pyplot() 

    
if selected == "Top Chart":
    if st.checkbox("Show Top Stores and Department Chart"):
            
    # Use sliders to filter by Year and Month
        Year = st.slider("Select Year", min_value=2010, max_value=2012, step=1)
        Month = st.slider("Select Month", min_value=1, max_value=12, step=1)
        
    # Filter the DataFrame based on selected Year and Month
        tab1,tab2 = st.tabs(["$\huge Store $", "$\huge Department $"])

        with tab1:
            result_df = df.groupby(['Store','Date_month', 'Date_year'])['Weekly_Sales'].sum().reset_index()
            filtered_df = result_df[(result_df['Date_year'] == Year) & (result_df['Date_month'] == Month)]
    
        # Create a Streamlit chart
            fig = px.pie(filtered_df, values='Weekly_Sales',
                        names='Store',
                        title=f'Top Stores based on Weekly Sales - {Year}-{Month}',
                        color_discrete_sequence=px.colors.sequential.Agsunset,
                        labels={'Weekly_Sales': 'Weekly Sales'})

            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            result_df = df.groupby(['Dept','Date_month', 'Date_year'])['Weekly_Sales'].sum().reset_index()
            filtered_df = result_df[(result_df['Date_year'] == Year) & (result_df['Date_month'] == Month)]
    
        # Create a Streamlit chart
            fig = px.pie(filtered_df, values='Weekly_Sales',
                        names='Dept',
                        title=f'Top Department based on Weekly Sales - {Year}-{Month}',
                        color_discrete_sequence=px.colors.sequential.Agsunset,
                        labels={'Weekly_Sales': 'Weekly Sales'})

            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        
if selected == "Weekly_Chart":
    if st.checkbox("Show Store, Department and year wise weekly_sales_price"):
        
    
    # Use sliders to filter by Year and Month
        selected_year = st.slider("Select Year", min_value=2010, max_value=2012, step=1)
        selected_store = st.slider("Select Store", min_value=1, max_value=45, step=1)
        selected_department = st.slider("Select Dept", min_value=1, max_value=99, step=1)

        # Filter the DataFrame based on user selection
        result_df = df.groupby(['Store', 'Dept', 'Date_year'])['Weekly_Sales'].sum().reset_index()
        filtered_df = result_df[(result_df['Date_year'] == selected_year) & 
                                (result_df['Store'] == selected_store) & 
                                (result_df['Dept'] == selected_department)]

        st.markdown(f"### :green[Weekly Sales for Year {selected_year}, Store {selected_store}, Department {selected_department}]")

        if not filtered_df.empty:
            fig = px.pie(filtered_df, values='Weekly_Sales',
                        names='Store',
                        title=f'Weekly Sales for Year {selected_year}, Store {selected_store}, Department {selected_department}',
                        labels={'Weekly_Sales': 'Weekly Sales'})

            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
                