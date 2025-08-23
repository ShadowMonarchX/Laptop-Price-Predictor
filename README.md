
# Laptop Price Predictor 💻

**Laptop Price Predictor** is a **web-based application** built using **Streamlit** and **Machine Learning**, which predicts the price of a laptop based on its configuration input by the user. The application supports multiple brands, models, and laptop types, including options for gaming, video editing, AI model training, and professional use.

---

## Features

* Predicts laptop price based on real-world configurations.
* Supports **major laptop brands**: Lenovo, HP, Dell, Apple, Asus, Samsung, MI, Microsoft, MSI, Acer.
* Allows selection of **specific models** within each brand.
* Supports **various use cases**:

  * Gaming
  * AI Model Training
  * Video Editing
  * Business and Professional Use
* Includes **advanced options**:

  * Operating System (Windows, macOS, Linux, Kali Linux, Red Hat, Ubuntu)
  * GPU Selection (NVIDIA, Intel, AMD)
  * Touchscreen and IPS display options
  * Weight, RAM, Storage, and Screen Size input
* Predicts price using a **stacking regressor** pipeline combining RandomForest, GradientBoosting, and Ridge Regression.
* Provides smooth and interactive **Streamlit UI**.

---

## Business Problem

Predict the price of laptops based on a dataset of **\~1300 laptop models**.

**Dataset Features:**

* Company Name
* Product Name / Model
* Laptop Type (Ultrabook, Gaming, Notebook, Video Editing, etc.)
* Screen Inches
* Screen Resolution
* CPU Model
* RAM
* Storage (HDD/SSD)
* GPU
* Operating System
* Laptop Weight
* Laptop Price

**Dataset Source:** [Kaggle - Laptop Prices](https://www.kaggle.com/ionaskel/laptop-prices)

**Machine Learning Problem:**

* **Regression** problem: Predict laptop price for a given configuration.

**Performance Metrics:**

1. **R² Score**
2. **Mean Absolute Error (MAE)**

---

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/Laptop-Price-Predictor.git
cd Laptop-Price-Predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the app in your browser and select the configuration of your laptop to get the predicted price.

---

## User Interface

**Enter Laptop Specifications:**

![Laptop Input UI](https://user-images.githubusercontent.com/63099028/183035027-a8e3e365-7d37-4ed1-93d3-02ecd44ff668.PNG)

**Predicted vs Actual Price Comparison:**

**Amazon Price:**
![Amazon Price](https://user-images.githubusercontent.com/63099028/180611808-28a90158-4f95-4199-b767-f53e584b7366.PNG)

**Predicted Price:**
![Predicted Price](https://user-images.githubusercontent.com/63099028/180611810-97d7b279-f1b0-4f62-9469-4f01b931e1f6.PNG)

---

## Future Enhancements

* Integrate **real-time market prices** from Amazon, Flipkart, and other e-commerce websites.
* Expand dataset to include more **laptop brands, models, and international variants**.
* Add **GPU benchmarking** and **performance recommendations** based on use-case selection.
* Deploy as a **cloud-based web app** for global access.
