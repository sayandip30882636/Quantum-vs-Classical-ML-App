import streamlit as st
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Quantum Device ---
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

# --- Multi-layer Quantum Circuit ---
@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.AngleEmbedding(x, wires=range(num_qubits))
    num_layers = len(params) // num_qubits
    for l in range(num_layers):
        for q in range(num_qubits):
            qml.RY(params[num_qubits*l + q], wires=q)
        qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

def variational_classifier(params, x):
    # Do NOT convert to float during training!
    return quantum_circuit(params, x)

def cost_function(params, X, y):
    predictions = [variational_classifier(params, xi) for xi in X]  # keep differentiable
    predictions = np.array(predictions)  # convert to PennyLane/NumPy array
    return np.mean((predictions - y)**2)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚛️ Quantum vs Classical ML Demo")
st.write("Compare a multi-layer quantum classifier with classical models on simple datasets.")

# Sidebar
st.sidebar.header("Dataset & Parameters")
dataset_type = st.sidebar.selectbox("Dataset", ["Moons", "Circles"])
n_samples = st.sidebar.slider("Number of samples", 50, 300, 150, 10)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2, 0.05)
epochs = st.sidebar.slider("Quantum Epochs", 20, 100, 50, 5)
learning_rate = st.sidebar.slider("Quantum Learning Rate", 0.01, 0.3, 0.1, 0.01)
num_layers = st.sidebar.slider("Quantum Layers", 1, 5, 3, 1)

# --- Generate dataset ---
if dataset_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)

# For quantum classifier: labels {-1, 1}
y_q = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_q, X_test_q = X_train, X_test
y_train_q, y_test_q = y_q[:len(X_train)], y_q[len(X_train):]

# --- Display dataset ---
st.subheader("Dataset")
fig, ax = plt.subplots()
ax.scatter(X[y==0][:,0], X[y==0][:,1], color='red', marker='o', label='Class 0')
ax.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', marker='x', label='Class 1')
ax.legend()
st.pyplot(fig)

# --- Train Models ---
if st.sidebar.button("Train Models"):
    st.subheader("Classical Models")
    classical_models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(kernel='rbf')
    }
    classical_results = {}
    for name, clf in classical_models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        classical_results[name] = acc
        st.write(f"**{name} Accuracy:** {acc:.2f}%")

    st.subheader("Quantum Classifier")
    num_params = num_layers * num_qubits
    params = np.random.uniform(0, np.pi, num_params)
    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    progress = st.progress(0)
    for epoch in range(epochs):
        params, _ = optimizer.step_and_cost(lambda p: cost_function(p, X_train_q, y_train_q), params)
        progress.progress(int((epoch+1)/epochs*100))

    # Convert QNode outputs to float for evaluation
    quantum_preds = np.sign([float(quantum_circuit(params, x)) for x in X_test_q])
    quantum_acc = np.mean(quantum_preds == y_test_q) * 100
    st.write(f"**Quantum Classifier Accuracy:** {quantum_acc:.2f}%")

    # --- Decision boundaries ---
    st.subheader("Decision Boundaries")
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z_classical = classical_models["Logistic Regression"].predict(grid).reshape(xx.shape)
    Z_quantum = np.array([float(quantum_circuit(params, p)) for p in grid]).reshape(xx.shape)

    fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.contourf(xx, yy, Z_classical, levels=[-1,0,1], cmap='RdBu', alpha=0.6)
    ax1.scatter(X[y==0][:,0], X[y==0][:,1], color='red', label='Class 0')
    ax1.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Class 1')
    ax1.set_title("Classical: Logistic Regression")

    ax2.contourf(xx, yy, Z_quantum, levels=[-1,0,1], cmap='RdBu', alpha=0.6)
    ax2.scatter(X[y==0][:,0], X[y==0][:,1], color='red', label='Class 0')
    ax2.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Class 1')
    ax2.set_title("Quantum Classifier")

    st.pyplot(fig2)

    st.subheader("Insights")
    st.write(f"- Quantum classifier with {num_layers} layers shows how quantum circuits can model complex patterns.")
    st.write(f"- Classical ML performs well on simple 2D datasets; quantum ML may excel on higher-dimensional or entangled data.")

else:
    st.info("Adjust parameters in the sidebar and click **Train Models** to start.")
