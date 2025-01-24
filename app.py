import streamlit as st
import pandas as pd

from model import customer_df, product_df, purchase_df, n_customers, n_products, add_customer, add_product, recommend_for_customer, recommend_for_product, reset
import utils

# LOAD MODEL
utils.reload(n_customers, n_products)

# USER INTERFACE (STREAMLIT APP)
st.title("Customer-Product Recommendation System")

# INSTRUCTIONS
st.sidebar.title("Instructions")
st.sidebar.write("1. Choose whether to input a customer or product.")
st.sidebar.write("2. For known entities, enter their name. For new ones, provide additional details.")

# HANDLE DIFFERENT INPUT
option = st.selectbox("Would you like to input a Customer or a Product?", ("Customer", "Product"))

# CUSTOMER
if option == "Customer":
    
    reset()
    
    customer_name = st.text_input("Enter Customer Name")
    
    if not customer_name.strip():  # Check for empty or whitespace-only input
        st.warning("Please enter a Customer Name.")
        
    # NEW CUSTOMER
    elif customer_name not in customer_df["CUSTOMER NAME"].values:
        st.write("New Customer Detected. Please input additional details:")
        customer_type = st.selectbox("Select Customer Type", ["End user", "Workshop"])  
        purchasing_power = st.slider("Select Purchasing Power", 0, 2000, step=100)
        
        if st.button("Get Recommendations"):
            
            # TEMPORARILY ADD TO DATAFRAME
            new_customer_id = len(customer_df)
            customer_df = add_customer(customer_df, customer_name, customer_type, purchasing_power, new_customer_id)
        
            my_model = utils.reload(n_customers, n_products)
            
            # SHOW RECOMMENDATIONS
            recommendations = recommend_for_customer(customer_df, new_customer_id)
            st.subheader("Recommended Products:")
            if recommendations:
                recommendations_table = [[product, f"{qty:.2f}"] for idx, (product, qty) in enumerate(recommendations)]
                st.table(pd.DataFrame(recommendations_table, columns=["Product Name", "Predicted Quantity"]))
            else:
                st.write("No recommendations available.")
    
    # EXISTING CUSTOMER        
    else:
        
        reset()
        
        customer_id = customer_df.loc[customer_df["CUSTOMER NAME"] == customer_name, "CUSTOMER_ID"].values[0]
        
        if st.button("Get Recommendations"):
            
            # SHOW PURCHASE HISTORY
            st.subheader("Purchase History:")
            utils.display_purchase_history(customer_id, "CUSTOMER ID", "DESCRIPTION", purchase_df, is_customer=True)

            # SHOW RECOMMENDATIONS
            recommendations = recommend_for_customer(customer_df, customer_id)
            st.subheader("Recommended Products:")
            if recommendations:
                recommendations_table = [[product, f"{qty:.2f}"] for idx, (product, qty) in enumerate(recommendations)]
                st.table(pd.DataFrame(recommendations_table, columns=["Product Name", "Predicted Quantity"]))
            else:
                st.write("No recommendations available.")

# PRODUCT
elif option == "Product":
    
    reset()
    
    product_name = st.text_input("Enter Product Name")
    
    if not product_name.strip():  # Check for empty or whitespace-only input
        st.warning("Please enter a valid Product Name.")
        
    # NEW PRODUCT
    elif product_name not in product_df["DESCRIPTION"].values:
        st.write("New Product Detected. Please input additional details:")
        category = st.selectbox("Select Category", ["Engine Oils & Fluids", "Filters", "Brake Components",
                                                    "Ignition Components", "Belts & Timing", "Cooling System",
                                                    "Suspension & Mountings", "Exterior & Accessories", "Seals & Gaskets",
                                                    "Electrical Components", "Miscellaneous", "Uncategorized"])
        price = st.slider("Select Product Price", 0, 550, step=10)
        
        if st.button("Get Recommendations"):
            
            # TEMPORARILY ADD TO DATAFRAME
            new_product_id = len(product_df)
            product_df = add_product(product_df, product_name, category, price, new_product_id)
        
            my_model = utils.reload(n_customers, n_products)
            
            # SHOW RECOMMENDATIONS
            recommendations = recommend_for_product(product_df, new_product_id)
            
            st.subheader("Recommended Customers:")
            if recommendations:
                recommendations_table = [[customer, f"{qty:.2f}"] for idx, (customer, qty) in enumerate(recommendations)]
                st.table(pd.DataFrame(recommendations_table, columns=["Customer Name", "Predicted Quantity"]))
            else:
                st.write("No recommendations available.")
    
    # EXISTING PRODUCT        
    else:
        
        reset()
        
        product_id = product_df.loc[product_df["DESCRIPTION"] == product_name, "PRODUCT_ID"].values[0]
        
        if st.button("Get Recommendations"):
            # SHOW PURCHASE HISTORY
            st.subheader("Purchase History:")
            utils.display_purchase_history(product_id, "PRODUCT ID", "CUSTOMER NAME", purchase_df, is_customer=False)

            # SHOW RECOMMENDATIONS
            recommendations = recommend_for_product(product_df, product_id)
            st.subheader("Recommended Customers:")
            if recommendations:
                recommendations_table = [[customer, f"{qty:.2f}"] for idx, (customer, qty) in enumerate(recommendations)]
                st.table(pd.DataFrame(recommendations_table, columns=["Customer Name", "Predicted Quantity"]))
            else:
                st.write("No recommendations available.")
                
            
