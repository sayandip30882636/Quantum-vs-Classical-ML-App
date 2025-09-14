An interactive Streamlit web application for comparing a quantum variational classifier against classical machine learning models (Logistic Regression, SVM) on 2D datasets.

This application provides a hands-on environment to visually explore how different machine learning models handle non-linearly separable data, offering insights into the potential strengths of quantum circuits. It's designed for students, researchers, and anyone interested in the practical side of quantum machine learning and the ongoing search for quantum advantage.

Link - https://quantum-vs-classical-ml-app.streamlit.app/

### 🚀 Key Features:

* **Interactive Controls:** Configure the entire experiment through a user-friendly sidebar without touching any code.
* **Dataset Customization:** Choose between "Moons" and "Circles" datasets. Adjust the number of samples to see how model performance scales and add noise to test the robustness of each algorithm.
* **Quantum Model Tuning:** Directly modify the quantum circuit's depth (layers) to increase its complexity. Tune training hyperparameters like epochs and learning rate to see their effect on the model's convergence.
* **Side-by-Side Comparison:** Train quantum and classical models simultaneously on the same data split for a fair comparison and instantly view their test accuracy scores.
* **Decision Boundary Visualization:** Go beyond metrics and see *how* each model learns to separate the data with clear, automatically generated visualizations of their decision boundaries.

### 🛠️ Getting Started

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

### 💻 Technologies Used:

* **Streamlit:** For building the interactive web application interface.
* **PennyLane:** For creating, simulating, and training the quantum variational circuit.
* **Scikit-learn:** For implementing and training the classical machine learning models.
* **NumPy:** For efficient numerical computation and data manipulation.
* **Matplotlib:** For generating the decision boundary and dataset plots.
"""
