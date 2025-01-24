import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import utils

class Dataset(Dataset):
    def __init__(self, x, y, customer_df, product_df):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.customer_type = torch.tensor(customer_df['CUSTOMER TYPE'].values, dtype=torch.long)
        self.purchasing_power = torch.tensor(customer_df['PURCHASING POWER'].values, dtype=torch.float32)
        
        self.product_category = torch.tensor(product_df['CATEGORY'].values, dtype=torch.long)
        self.product_price = torch.tensor(product_df['U/PRICE'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        customer_id, product_id = self.x[idx]
        return (
            customer_id,
            product_id,
            self.customer_type[customer_id],
            self.purchasing_power[customer_id],
            self.product_category[product_id],
            self.product_price[product_id],
            self.y[idx],
        )

scaler = MinMaxScaler()

customer_df = pd.read_csv("customer_data.csv")
customer_encoder = LabelEncoder()
customer_df['CUSTOMER_ID'] = customer_encoder.fit_transform(customer_df['CUSTOMER NAME'])
customer_df['CUSTOMER TYPE'] = customer_df['CUSTOMER TYPE'].astype('category').cat.codes
customer_df['PURCHASING POWER'] = scaler.fit_transform(customer_df[['PURCHASING POWER']])

product_df = pd.read_csv("product_data.csv")
product_encoder = LabelEncoder()
product_df['PRODUCT_ID'] = product_encoder.fit_transform(product_df['DESCRIPTION'])
product_df['CATEGORY'] = product_df['CATEGORY'].astype('category').cat.codes
product_df['U/PRICE'] = scaler.fit_transform(product_df[['U/PRICE']])

purchase_df = pd.read_csv("purchase_data.csv")
purchase_df['CUSTOMER ID'] = customer_encoder.fit_transform(purchase_df['CUSTOMER NAME'])
purchase_df['PRODUCT ID'] = customer_encoder.fit_transform(purchase_df['DESCRIPTION'])
x = purchase_df[['CUSTOMER ID', 'PRODUCT ID']].values
y = purchase_df['QTY'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# DATA PREPARATION

BATCH_SIZE = 32

train_df = Dataset(x_train, y_train, customer_df, product_df)
train_loader = DataLoader(train_df, BATCH_SIZE, shuffle=True)

test_df = Dataset(x_test, y_test, customer_df, product_df)
test_loader = DataLoader(test_df, BATCH_SIZE)

n_customers = len(customer_df)
n_products = len(product_df)

type_dim = customer_df['CUSTOMER TYPE'].nunique()   # 2
category_dim = product_df['CATEGORY'].nunique()     # 12
model = utils.NCFModel(n_customers, n_products, embedding_dim=8, n_features=2, 
                       type_dim=type_dim, category_dim=category_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 15
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(model, train_loader, optimizer, criterion)
    utils.test(model, test_loader, criterion)
    
original_customer_embedding = model.customer_embedding
original_product_embedding = model.product_embedding

# RECOMMEND FUNCTION   
def recommend_for_customer(customer_df, customer_id, top_n=10):
    if customer_id >= n_customers:
        raise ValueError(f"Customer ID {customer_id} is out of range.")

    customer_type = torch.tensor([customer_df.loc[customer_id, 'CUSTOMER TYPE']] * n_products, dtype=torch.long)
    purchasing_power = torch.tensor([customer_df.loc[customer_id, 'PURCHASING POWER']] * n_products, dtype=torch.float32)
    
    product_ids = torch.arange(n_products)
    product_category = torch.tensor(product_df['CATEGORY'].values, dtype=torch.long)
    product_price = torch.tensor(product_df['U/PRICE'].values, dtype=torch.float32)
    
    with torch.no_grad():
        pred_qty = model(torch.tensor([customer_id] * n_products), product_ids, customer_type, 
                         purchasing_power, product_category, product_price).squeeze()
    recommendations = pred_qty.argsort(descending=True)[:top_n]
    return [(product_encoder.inverse_transform([pid.item()])[0], pred_qty[pid].item()) for pid in recommendations]

def recommend_for_product(product_df, product_id, top_n=10):
    if product_id >= n_products:
        raise ValueError(f"Product ID {product_id} is out of range.")
    
    product_category = torch.tensor([product_df.loc[product_id, 'CATEGORY']] * n_customers, dtype=torch.long)
    product_price = torch.tensor([product_df.loc[product_id, 'U/PRICE']] * n_customers, dtype=torch.float32)
    
    customer_ids = torch.arange(n_customers)
    customer_type = torch.tensor(customer_df['CUSTOMER TYPE'].values, dtype=torch.long)
    purchasing_power = torch.tensor(customer_df['PURCHASING POWER'].values, dtype=torch.float32)
    
    with torch.no_grad():
        pred_qty = model(customer_ids, torch.tensor([product_id] * n_customers), customer_type, 
                         purchasing_power, product_category, product_price).squeeze()
    recommendations = pred_qty.argsort(descending=True)[:top_n]
    return [(customer_encoder.inverse_transform([cid.item()])[0], pred_qty[cid].item()) for cid in recommendations]

# ADD ENTITY FUNCTION
def add_customer(customer_df, customer_name, customer_type, purchasing_power, new_customer_id):
    
    global n_customers
    n_customers += 1
    
    temp_n = model.customer_embedding.num_embeddings
    
    if customer_type=="End user":
        encoded_type = 0
    else:
        encoded_type = 1
        
    new_customer = pd.DataFrame([{
        "CUSTOMER NAME": customer_name,
        "CUSTOMER TYPE": encoded_type,
        "PURCHASING POWER": purchasing_power/2073,
        "CUSTOMER_ID": new_customer_id
    }])
    customer_df = pd.concat([customer_df, new_customer], ignore_index=True)

    new_embedding_layer = nn.Embedding(n_customers, model.customer_embedding.embedding_dim)
    with torch.no_grad():
        new_embedding_layer.weight[:temp_n] = model.customer_embedding.weight
        new_embedding_layer.weight[temp_n] = torch.randn(model.customer_embedding.embedding_dim)
    model.customer_embedding = new_embedding_layer
    torch.save(model.state_dict(), "model.pth")
    
    return customer_df
    
def add_product(product_df, product_name, category, price, new_product_id):
    
    global n_products
    n_products += 1
    
    temp_n = model.product_embedding.num_embeddings

    categories = [
        "Belts & Timing", "Brake Components", "Cooling System",
        "Electrical Components", "Engine Oils & Fluids", "Exterior & Accessories",
        "Filters", "Ignition Components", "Miscellaneous", "Seals & Gaskets",
        "Suspension & Mountings", "Uncategorized"
    ]
    
    encoded_category = categories.index(category)

    new_product = pd.DataFrame([{
        "DESCRIPTION": product_name,
        "CATEGORY": encoded_category,
        "U/PRICE": price/550,
        "PRODUCT_ID": new_product_id
    }])
    product_df = pd.concat([product_df, new_product], ignore_index=True)

    new_embedding_layer = nn.Embedding(n_products, model.product_embedding.embedding_dim)
    with torch.no_grad():
        new_embedding_layer.weight[:temp_n] = model.product_embedding.weight
        new_embedding_layer.weight[temp_n] = torch.randn(model.product_embedding.embedding_dim)
    model.product_embedding = new_embedding_layer
    torch.save(model.state_dict(), "model.pth")

    return product_df
    
# RESET DATASET & EMBEDDING    
def reset():
    global customer_df, product_df, n_customers, n_products
    
    customer_df = pd.read_csv("customer_data.csv")
    customer_df['CUSTOMER_ID'] = customer_encoder.fit_transform(customer_df['CUSTOMER NAME'])
    customer_df['CUSTOMER TYPE'] = customer_df['CUSTOMER TYPE'].astype('category').cat.codes
    customer_df['PURCHASING POWER'] = scaler.fit_transform(customer_df[['PURCHASING POWER']])

    product_df = pd.read_csv("product_data.csv")
    product_df['PRODUCT_ID'] = product_encoder.fit_transform(product_df['DESCRIPTION'])
    product_df['CATEGORY'] = product_df['CATEGORY'].astype('category').cat.codes
    product_df['U/PRICE'] = scaler.fit_transform(product_df[['U/PRICE']])
    
    n_customers = len(customer_df)
    n_products = len(product_df)
    model.customer_embedding = original_customer_embedding
    model.product_embedding = original_product_embedding
    
    utils.reload(n_customers, n_products)

# Save the trained model
torch.save(model.state_dict(), "model.pth")