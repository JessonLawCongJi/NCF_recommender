import pandas as pd
import utils
    
file1 = "cash listing summary-csv.csv"
file2 = "invoice listing summary-csv.csv"

header1 = r'"(CS-[^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"'
header2 = r'"(IV-[^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"'
product_pattern = r'"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"'

# READ DATASET
cash_df = utils.clean_dataset(file1, header1, product_pattern)
invoice_df = utils.clean_dataset(file2, header2, product_pattern)

# COMBINE DATASET
data = pd.concat([cash_df, invoice_df], ignore_index=True)

categories = {
    # Engine Oils & Fluids
    "engine oil|eng oil|oil treatment|lubricant|coolant|brake fluid|gear oil|atf oil|power stg fluid": "Engine Oils & Fluids",
    
    # Filters
    "oil filter|air filter|cabin filter|fuel filter": "Filters",
    
    # Brake Components
    "brake|disc pad|brake shoe|disc rotor|brake pump|caliper|abs": "Brake Components",
    
    # Ignition Components
    "spark plug|plug cable|knock sensor|ignition": "Ignition Components",
    
    # Belts & Timing
    "fan belt|timing belt|belt tensioner": "Belts & Timing",
    
    # Cooling System
    "coolant|radiator|water pump|thermostat|fan switch": "Cooling System",
    
    # Suspension & Mountings
    "bush|joint|arm|mounting|shock absorber|ball joint|tie rod|idler arm|center rod|axle|spring|lower arm|upper arm|engine mtg|gear box mtg|longshaft": "Suspension & Mountings",
    
    # Exterior & Accessories
    "wiper|blade|mirror|mudflap|door|tail lamp|light|bulb|washer nozzle|handle|sticker": "Exterior & Accessories",
    
    # Seals & Gaskets
    "seal|gasket|o-ring|half moon|oil cap o ring|timing cover|ribbon sealer|cock|valve cover": "Seals & Gaskets",
    
    # Electrical Components
    "relay|switch|sensor|knock sensor|power window|thermostat|fan switch|w/s sealant": "Electrical Components",
    
    # Miscellaneous
    "additive|treatment|grease|lubricant|hose|fuel hose|clip|pipe|nut|bolt|washer|rubber|anti-rust|grinding compound": "Miscellaneous",
}

customer_type = {
    "CS": "End user",
    "IV": "Workshop",
}

data["CATEGORY"] = data["DESCRIPTION"].apply(lambda row: utils.categorise(row, categories))
data["CUSTOMER TYPE"] = data["DOC NO"].apply(lambda row: utils.customer(row, customer_type))
data["CUSTOMER NAME"] = data.apply(utils.rename_customer, axis=1)

data = utils.calculate_purchasing_power(data)

customer_df = data[["CUSTOMER NAME", "CUSTOMER TYPE", "PURCHASING POWER"]].drop_duplicates(subset=["CUSTOMER NAME"])
product_df = data[["DESCRIPTION", "CATEGORY","U/PRICE"]].drop_duplicates(subset=["DESCRIPTION"])
purchase_df = data[["CUSTOMER NAME", "DESCRIPTION", "QTY"]]

# SAVE TO DIFFERENT DATASET
customer_df.to_csv("customer_data.csv", index=False)
product_df.to_csv("product_data.csv", index=False)
purchase_df.to_csv("purchase_data.csv", index=False)