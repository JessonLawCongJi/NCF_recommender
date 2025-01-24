import re
import csv

input_file = "cash listing summary-csv.csv"

doc_header_pattern = re.compile(r'"(CS-[^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"')
product_pattern = re.compile(r'"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"')

last_doc_no = last_date = last_customer_id = last_customer_name = None

def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

with open(input_file, mode='r', encoding='utf-8') as infile:
    input_data = infile.read()
    
output_file = 'cash_clean.csv'

with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)

    writer.writerow(["DOC NO", "DATE", "CUSTOMER ID", "CUSTOMER NAME", "STOCK CODE", "DESCRIPTION", "QTY", "U/PRICE", "AMOUNT"])
    for line in input_data.splitlines():
        doc_match = doc_header_pattern.search(line)
        if doc_match:
            last_doc_no, last_date, last_customer_id, last_customer_name = doc_match.groups()
            continue
        
        product_match = product_pattern.search(line)
        if product_match:
            stock_code, description, qty, u_price, amount = product_match.groups()[1:-1]
            if is_number(qty) and is_number(amount):
                writer.writerow([last_doc_no, last_date, last_customer_id, last_customer_name, stock_code, description, qty, u_price, amount])