# Customer-Product Recommendation System

This repository contains the implementation of a **Customer-Product Recommendation System** developed using Neural Collaborative Filtering (NCF). The system leverages customer and product attributes to deliver personalized and accurate recommendations. It is specifically designed for the auto parts industry but is flexible enough to be applied to other domains.

## Key Features

- **Neural Collaborative Filtering (NCF)**: Utilizes embedding layers to capture customer and product features.
- **Dynamic Entity Updates**: Supports addition of new customers and products without requiring complete model retraining.
- **Streamlit-Based Interface**: Provides a user-friendly interface for input and output handling.
- **Scalable and Modular Design**: Allows easy integration of new features and seamless adaptability.

## Project Objectives

1. Implement a neural collaborative filtering-based recommendation system for customer-product matching.
2. Analyze sales data and generate personalized recommendations.
3. Evaluate system performance using metrics like precision, recall, and RMSE.

## System Overview

### Architecture
The system is built with a modular architecture to ensure scalability and maintainability. Key components include:

1. **Recommendation Engine**: Built using PyTorch for NCF implementation.
2. **Data Preprocessing**: Handles encoding of features, normalization, and train/test splitting.
3. **User Interface**: Built using Streamlit to provide real-time interaction.
4. **Dynamic Updates**: Adds new entities dynamically without retraining the model.

### Workflow
1. Load and preprocess data.
2. Train the NCF model with customer and product embeddings.
3. Use the trained model to predict recommendations based on user input.
4. Display recommendations through the Streamlit interface.

## Dataset

1. **Customer Data**: Contains customer attributes like type and purchasing power.
2. **Product Data**: Includes product descriptions, categories, and unit prices.
3. **Purchase Data**: Represents historical transactions with customer-product pairs and quantities.

## Libraries Used

- **PyTorch**: For implementing and training the NCF model.
- **pandas**: For data manipulation and preprocessing.
- **scikit-learn**: For train/test splitting and evaluation.
- **Streamlit**: For building an interactive user interface.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/customer-product-recommendation-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-product-recommendation-system
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the application using Streamlit.
2. Choose between customer-based or product-based recommendations on the homepage.
3. Enter input details (e.g., new/existing customer or product).
4. View personalized recommendations generated by the system.
