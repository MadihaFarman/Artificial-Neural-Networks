import numpy as np

def sigmoid(z):
    # squashes any number into the range (0, 1)
    return 1 / (1 + np.exp(-z))

# ---------- one training example ----------
x1 = 0.5
x2 = 0.8
y  = 1          # the true answer we want the network to predict

# ---------- Layer 1 parameters (3 neurons, each receives x1 and x2) ----------
w1_11, w1_12, b1_1 =  0.2, -0.3,  0.10    # neuron 1
w1_21, w1_22, b1_2 =  0.4,  0.1, -0.10    # neuron 2
w1_31, w1_32, b1_3 = -0.5,  0.6,  0.05    # neuron 3

# ---------- Layer 2 parameters (2 neurons, each receives a1_1, a1_2, a1_3) ----------
w2_11, w2_12, w2_13, b2_1 =  0.3, -0.2, 0.5, 0.00   # neuron 1
w2_21, w2_22, w2_23, b2_2 = -0.4,  0.6, 0.1, 0.10   # neuron 2

# ---------- Output layer parameters (1 neuron, receives a2_1, a2_2) ----------
w3_1, w3_2, b3 = 0.7, -0.5, 0.20

# ---------- learning rate ----------
lr = 0.5

# ----- Layer 1, neuron 1 -----
z1_1 = w1_11 * x1 + w1_12 * x2 + b1_1
a1_1 = sigmoid(z1_1)
print(f"L1 neuron 1 | IN : x1={x1}, x2={x2} | weights: w1_11={w1_11:.4f}, w1_12={w1_12:.4f} | bias: b1_1={b1_1:.4f}")
print(f"            | OUT: z1_1={z1_1:.4f}  ->  a1_1={a1_1:.4f}")

# ----- Layer 1, neuron 2 -----
z1_2 = w1_21 * x1 + w1_22 * x2 + b1_2
a1_2 = sigmoid(z1_2)
print(f"L1 neuron 2 | IN : x1={x1}, x2={x2} | weights: w1_21={w1_21:.4f}, w1_22={w1_22:.4f} | bias: b1_2={b1_2:.4f}")
print(f"            | OUT: z1_2={z1_2:.4f}  ->  a1_2={a1_2:.4f}")

# ----- Layer 1, neuron 3 -----
z1_3 = w1_31 * x1 + w1_32 * x2 + b1_3
a1_3 = sigmoid(z1_3)
print(f"L1 neuron 3 | IN : x1={x1}, x2={x2} | weights: w1_31={w1_31:.4f}, w1_32={w1_32:.4f} | bias: b1_3={b1_3:.4f}")
print(f"            | OUT: z1_3={z1_3:.4f}  ->  a1_3={a1_3:.4f}")

# ----- Layer 2, neuron 1 -----
z2_1 = w2_11 * a1_1 + w2_12 * a1_2 + w2_13 * a1_3 + b2_1
a2_1 = sigmoid(z2_1)
print(f"L2 neuron 1 | IN : a1_1={a1_1:.4f}, a1_2={a1_2:.4f}, a1_3={a1_3:.4f}")
print(f"            | weights: w2_11={w2_11:.4f}, w2_12={w2_12:.4f}, w2_13={w2_13:.4f} | bias: b2_1={b2_1:.4f}")
print(f"            | OUT: z2_1={z2_1:.4f}  ->  a2_1={a2_1:.4f}")

# ----- Layer 2, neuron 2 -----
z2_2 = w2_21 * a1_1 + w2_22 * a1_2 + w2_23 * a1_3 + b2_2
a2_2 = sigmoid(z2_2)
print(f"L2 neuron 2 | IN : a1_1={a1_1:.4f}, a1_2={a1_2:.4f}, a1_3={a1_3:.4f}")
print(f"            | weights: w2_21={w2_21:.4f}, w2_22={w2_22:.4f}, w2_23={w2_23:.4f} | bias: b2_2={b2_2:.4f}")
print(f"            | OUT: z2_2={z2_2:.4f}  ->  a2_2={a2_2:.4f}")

# ----- Output neuron -----
z3 = w3_1 * a2_1 + w3_2 * a2_2 + b3
y_hat = sigmoid(z3)
print(f"Output neuron | IN : a2_1={a2_1:.4f}, a2_2={a2_2:.4f} | weights: w3_1={w3_1}, w3_2={w3_2} | bias: b3={b3}")
print(f"              | OUT: z3={z3:.4f}  ->  y_hat={y_hat:.4f}")
print()
print(f"Prediction y_hat = {y_hat:.4f},  true answer y = {y}")

# ----- Loss -----
loss = 0.5 * (y - y_hat) ** 2
print(f"Loss before training = {loss:.6f}")

# ----- Output layer gradients -----
dL_dyhat = y_hat - y
delta3 = dL_dyhat * y_hat * (1 - y_hat)

dL_dw3_1 = delta3 * a2_1
dL_dw3_2 = delta3 * a2_2
dL_db3   = delta3

print(f"delta3   = {delta3:.6f}")
print(f"dL/dw3_1 = {dL_dw3_1:.6f}")
print(f"dL/dw3_2 = {dL_dw3_2:.6f}")
print(f"dL/db3   = {dL_db3:.6f}")

# ----- Hidden layer 2 gradients -----
# error flowing back into layer 2 outputs
dL_da2_1 = delta3 * w3_1
dL_da2_2 = delta3 * w3_2

# through the sigmoid of each layer-2 neuron
delta2_1 = dL_da2_1 * a2_1 * (1 - a2_1)
delta2_2 = dL_da2_2 * a2_2 * (1 - a2_2)

# parameter gradients
dL_dw2_11 = delta2_1 * a1_1
dL_dw2_12 = delta2_1 * a1_2
dL_dw2_13 = delta2_1 * a1_3
dL_db2_1   = delta2_1

dL_dw2_21 = delta2_2 * a1_1
dL_dw2_22 = delta2_2 * a1_2
dL_dw2_23 = delta2_2 * a1_3
dL_db2_2   = delta2_2

print(f"delta2_1 = {delta2_1:.6f}, delta2_2 = {delta2_2:.6f}")
print(f"dL_dw2_11={dL_dw2_11:.6f}, dL_dw2_12={dL_dw2_12:.6f}, dL_dw2_13={dL_dw2_13:.6f}, dL_db2_1={dL_db2_1:.6f}")
print(f"dL_dw2_21={dL_dw2_21:.6f}, dL_dw2_22={dL_dw2_22:.6f}, dL_dw2_23={dL_dw2_23:.6f}, dL_db2_2={dL_db2_2:.6f}")

# ----- Hidden layer 1 gradients -----
# error flowing back into layer 1 outputs (sum over the two paths)
dL_da1_1 = delta2_1 * w2_11 + delta2_2 * w2_21
dL_da1_2 = delta2_1 * w2_12 + delta2_2 * w2_22
dL_da1_3 = delta2_1 * w2_13 + delta2_2 * w2_23

# through the sigmoid of each layer-1 neuron
delta1_1 = dL_da1_1 * a1_1 * (1 - a1_1)
delta1_2 = dL_da1_2 * a1_2 * (1 - a1_2)
delta1_3 = dL_da1_3 * a1_3 * (1 - a1_3)

# parameter gradients
dL_dw1_11 = delta1_1 * x1
dL_dw1_12 = delta1_1 * x2
dL_db1_1   = delta1_1

dL_dw1_21 = delta1_2 * x1
dL_dw1_22 = delta1_2 * x2
dL_db1_2   = delta1_2

dL_dw1_31 = delta1_3 * x1
dL_dw1_32 = delta1_3 * x2
dL_db1_3   = delta1_3

print(f"delta1_1 = {delta1_1:.6f}, delta1_2 = {delta1_2:.6f}, delta1_3 = {delta1_3:.6f}")
print(f"dL_dw1_11={dL_dw1_11:.6f}, dL_dw1_12={dL_dw1_12:.6f}, dL_db1_1={dL_db1_1:.6f}")
print(f"dL_dw1_21={dL_dw1_21:.6f}, dL_dw1_22={dL_dw1_22:.6f}, dL_db1_2={dL_db1_2:.6f}")
print(f"dL_dw1_31={dL_dw1_31:.6f}, dL_dw1_32={dL_dw1_32:.6f}, dL_db1_3={dL_db1_3:.6f}")

# ----- Gradient descent: update every parameter -----
# Output layer
w3_1 = w3_1 - lr * dL_dw3_1
w3_2 = w3_2 - lr * dL_dw3_2
b3   = b3   - lr * dL_db3

# Hidden layer 2
w2_11 = w2_11 - lr * dL_dw2_11
w2_12 = w2_12 - lr * dL_dw2_12
w2_13 = w2_13 - lr * dL_dw2_13
b2_1   = b2_1   - lr * dL_db2_1
w2_21 = w2_21 - lr * dL_dw2_21
w2_22 = w2_22 - lr * dL_dw2_22
w2_23 = w2_23 - lr * dL_dw2_23
b2_2   = b2_2   - lr * dL_db2_2

# Hidden layer 1
w1_11 = w1_11 - lr * dL_dw1_11
w1_12 = w1_12 - lr * dL_dw1_12
b1_1   = b1_1   - lr * dL_db1_1
w1_21 = w1_21 - lr * dL_dw1_21
w1_22 = w1_22 - lr * dL_dw1_22
b1_2   = b1_2   - lr * dL_db1_2
w1_31 = w1_31 - lr * dL_dw1_31
w1_32 = w1_32 - lr * dL_dw1_32
b1_3   = b1_3   - lr * dL_db1_3

print("All 20 parameters updated!")

# ----- Forward propagation again with the new parameters -----
a1_1 = sigmoid(w1_11 * x1 + w1_12 * x2 + b1_1)
a1_2 = sigmoid(w1_21 * x1 + w1_22 * x2 + b1_2)
a1_3 = sigmoid(w1_31 * x1 + w1_32 * x2 + b1_3)

a2_1 = sigmoid(w2_11 * a1_1 + w2_12 * a1_2 + w2_13 * a1_3 + b2_1)
a2_2 = sigmoid(w2_21 * a1_1 + w2_22 * a1_2 + w2_23 * a1_3 + b2_2)

y_hat_new = sigmoid(w3_1 * a2_1 + w3_2 * a2_2 + b3)
loss_new = 0.5 * (y - y_hat_new) ** 2

print(f"Before: y_hat = {y_hat:.4f}, loss = {loss:.6f}")
print(f"After : y_hat = {y_hat_new:.4f}, loss = {loss_new:.6f}")

# ----- Training loop (same math, numpy arrays) -----
x = np.array([0.5, 0.8])
y = 1

W1 = np.array([[ 0.2, -0.3], [0.4, 0.1], [-0.5, 0.6]]);  b1 = np.array([0.10, -0.10, 0.05])
W2 = np.array([[ 0.3, -0.2, 0.5], [-0.4, 0.6, 0.1]]);    b2 = np.array([0.00, 0.10])
W3 = np.array([0.7, -0.5]);                              b3 = 0.20
lr = 0.5

for epoch in range(1, 501):
    # ---- forward ----
    a1 = sigmoid(W1 @ x + b1)
    a2 = sigmoid(W2 @ a1 + b2)
    y_hat = sigmoid(W3 @ a2 + b3)
    loss = 0.5 * (y - y_hat) ** 2

    # ---- backward ----
    delta3 = (y_hat - y) * y_hat * (1 - y_hat)
    delta2 = (delta3 * W3) * a2 * (1 - a2)
    delta1 = (W2.T @ delta2) * a1 * (1 - a1)

    # ---- update ----
    W3 = W3 - lr * delta3 * a2;            b3 = b3 - lr * delta3
    W2 = W2 - lr * np.outer(delta2, a1);   b2 = b2 - lr * delta2
    W1 = W1 - lr * np.outer(delta1, x);    b1 = b1 - lr * delta1

    if epoch in (1, 2, 10, 50, 100, 200, 500):
        print(f"epoch {epoch:3d} | y_hat = {y_hat:.4f} | loss = {loss:.6f}")
