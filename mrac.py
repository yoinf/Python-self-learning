import numpy as np
import matplotlib.pyplot as plt

# Desired reference model dynamics (ideal vessel response)
def reference_model(x_ref, r, dt):
    # Simple 1st-order system: x_dot = -a*x + b*r
    a, b = 0.5, 1.0
    dx = -a * x_ref + b * r
    return x_ref + dx * dt

# Actual vessel dynamics (unknown torque behavior)
def vessel_dynamics(x, u, dt):
    # Non-ideal system: different parameters + disturbance
    a, b = 0.8, 0.7
    disturbance = 0.05 * np.sin(0.5 * x)
    dx = -a * x + b * u + disturbance
    return x + dx * dt

# Adaptive controller (MRAC-like)
def adaptive_control(x, x_ref, u, theta, gamma, dt):
    # Tracking error
    e = x - x_ref

    # Control law: u = -theta * x + r
    r = np.sin(0.05 * t)  # reference input (desired command)
    u = -theta * x + r

    # Adaptive law: update theta to reduce error
    dtheta = -gamma * e * x
    theta += dtheta * dt

    return u, theta, e

# Simulation parameters
T = 300
N = 1000

dt = T/N
t = 0

# Initialize states
x = 0.0          # actual vessel state
x_ref = 0.0      # reference model state
theta = 0.0      # adaptive parameter

# Logging
X, X_ref, U, Theta, E = [], [], [], [], []

# Simulation loop
for k in range(N):
    t = k * dt
    r = np.sin(0.05 * t)

    # Reference model update
    x_ref = reference_model(x_ref, r, dt)

    # Adaptive control
    u, theta, e = adaptive_control(x, x_ref, r, theta, gamma=0.1, dt=dt)

    # Vessel dynamics update
    x = vessel_dynamics(x, u, dt)

    # Log
    X.append(x)
    X_ref.append(x_ref)
    U.append(u)
    Theta.append(theta)
    E.append(e)

# Plot results
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(X, label='Vessel')
plt.plot(X_ref, '--', label='Reference')
plt.legend()
plt.title('Adaptive Control of Vessel Dynamics')

plt.subplot(3,1,2)
plt.plot(U, label='Control Input (RPM equivalent)')
plt.legend()

plt.subplot(3,1,3)
plt.plot(Theta, label='Adaptive Parameter')
plt.plot(E, label='Tracking Error')
plt.legend()
plt.tight_layout()
plt.show()


