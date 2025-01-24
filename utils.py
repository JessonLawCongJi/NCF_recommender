import re
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd

# DATA
def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False    

def clean_dataset(input_file, header_pattern, product_pattern):
    
    header_pattern = re.compile(header_pattern)
    product_pattern = re.compile(product_pattern)
    
    last_doc_no = last_date = last_customer_id = last_customer_name = None
    
    rows = []
    
    with open(input_file, mode='r', encoding='utf-8') as infile:
        for line in infile:
            doc_match = header_pattern.search(line)
            if doc_match:
                last_doc_no, last_date, last_customer_id, last_customer_name = doc_match.groups()
                continue

            product_match = product_pattern.search(line)
            if product_match:
                stock_code, description, qty, u_price, amount = product_match.groups()[1:-1]

                if is_number(qty) and is_number(amount):
                    rows.append([
                        last_doc_no, last_date, last_customer_id, last_customer_name,
                        stock_code, description, qty, u_price, amount
                    ])

    df = pd.DataFrame(rows, columns=[
        "DOC NO", "DATE", "CUSTOMER ID", "CUSTOMER NAME","STOCK CODE", "DESCRIPTION", "QTY", "U/PRICE", "AMOUNT"
    ])

    return df

def categorise(row, categories):
    for pattern, category in categories.items():
        if re.search(pattern, row, re.IGNORECASE):
            return category
    return "Uncategorized"    

def customer(row, customer_type):
    for pattern, type in customer_type.items():
        if re.search(pattern, row, re.IGNORECASE):
            return type
    return "Walk-in"

def rename_customer(row):
    if row["CUSTOMER NAME"] in ["CASH", "KOGI"]:
        return f"Customer_{row['DOC NO']}"
    return row["CUSTOMER NAME"]

def calculate_purchasing_power(data, amount_col="AMOUNT", doc_col="DOC NO", customer_col="CUSTOMER NAME"):
    
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce").fillna(0)

    purchasing_power = data.groupby(customer_col).apply(
        lambda x: round(x[amount_col].sum() / x[doc_col].nunique(), 2) if x[doc_col].nunique() > 0 else 0
    ).reset_index(name="PURCHASING POWER")

    data = data.merge(purchasing_power, on=customer_col, how="left")

    return data

# MODEL
class NCFModel(nn.Module):
    def __init__(self, n_customers, n_products, embedding_dim, n_features, type_dim, category_dim):
        super(NCFModel, self).__init__()
        self.customer_embedding = nn.Embedding(n_customers, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)
        
        self.customer_type_embedding = nn.Embedding(type_dim, embedding_dim)
        self.purchasing_power_embedding = nn.Linear(1, embedding_dim)
        
        self.product_category_embedding = nn.Embedding(category_dim, embedding_dim)
        self.product_price_embedding = nn.Linear(1, embedding_dim)
        
        # Feature layers
        self.fc1 = nn.Linear(embedding_dim * 6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, customer_id, product_id, customer_type, purchasing_power, product_category, product_price):
        customer_emb = self.customer_embedding(customer_id)
        product_emb = self.product_embedding(product_id)
        type_emb = self.customer_type_embedding(customer_type)
        power_emb = self.purchasing_power_embedding(purchasing_power.unsqueeze(-1))
        category_emb = self.product_category_embedding(product_category)
        price_emb = self.product_price_embedding(product_price.unsqueeze(-1))

        # Combine embeddings
        x = torch.cat([customer_emb, product_emb, type_emb, power_emb, category_emb, price_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x).squeeze()
    
def get_batch_accuracy(output, y,threshold):
    correct = torch.abs(output - y) < threshold  
    return correct.sum().item() / len(y)

def train(model, train_loader, optimizer, criterion):
    model.train()  # Set model to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for customer_id, product_id, customer_type, purchasing_power, product_category, product_price, qty in train_loader:
        optimizer.zero_grad()
        predictions = model(customer_id, product_id, customer_type, purchasing_power, product_category, product_price).squeeze()
        
        loss = criterion(predictions, qty)
        loss.backward()
        optimizer.step()
        
        batch_accuracy = get_batch_accuracy(predictions, qty, threshold=0.7)
        total_correct += batch_accuracy * len(qty)
        total_samples += len(qty)

        total_loss += loss.item()
    
    train_accuracy = (total_correct / total_samples)
        
    print(f"Train Loss: {loss.item()}, Accuracy: {train_accuracy}")
    
def test(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient tracking for testing
        for customer_id, product_id, customer_type, purchasing_power, product_category, product_price, qty in test_loader:
            predictions = model(customer_id, product_id, customer_type, purchasing_power, product_category, product_price).squeeze()
            
            loss = criterion(predictions, qty)
            
            batch_accuracy = get_batch_accuracy(predictions, qty, threshold=0.7)
            total_correct += batch_accuracy * len(qty)
            total_samples += len(qty)

            total_loss += loss.item()
    
    test_accuracy = total_correct / total_samples
    print(f"Test Loss: {total_loss / len(test_loader)}, Accuracy: {test_accuracy}")

# USER INTERFACE    
def display_purchase_history(entity_id, entity_column, name_column, purchase_df, is_customer=True):
    history = purchase_df[purchase_df[entity_column] == entity_id]
    if history.empty:
        st.write("No purchase history available.")
    else:
        col_name = "Product Name" if is_customer else "Customer Name"
        table_data = [
            [row[name_column], row["QTY"]] for _, row in history.iterrows()
        ]
        st.table(pd.DataFrame(table_data, columns=[col_name, "Quantity"]))
    
def reload(n_customers, n_products):
    model = NCFModel(n_customers, n_products, embedding_dim=8, n_features=2, type_dim=2, category_dim=12)
    
    try:
        model.load_state_dict(torch.load("model.pth"))
    except RuntimeError as e:
        print(f"Warning: {e}")
        # Load the state dict with an option to ignore missing or mismatched keys
        model_dict = model.state_dict()
        pretrained_dict = torch.load("model.pth")
        # Filter out the weights that don't match
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
    return model