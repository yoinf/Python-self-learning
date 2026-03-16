"""
Adaptive 4‑parameter thruster controller for a 3‑DOF surface vessel
-------------------------------------------------------------------
Controls two azimuth thrusters with parameters:
  p = [f1, f2, d1, d2]  (forces [N] ~ RPM-scaled, and angles [rad])
so that the produced body‑wrench tau = [X, Y, N]^T tracks a desired tau_des.

Key ideas
- Outer loop computes desired body wrench from velocity tracking (PD on v).
- Inner adaptive layer updates [f1,f2,d1,d2] by projected gradient descent on
    J(p) = 1/2 * || B(d1,d2) @ [f1,f2] - tau_des ||^2  + regularizers
- Thruster geometry is configurable; constraints on forces and angles enforced.
- Simple 3‑DOF rigid‑body + linear damping simulation included to visualize behavior.

This is "model‑light": we do NOT need accurate hydrodynamics to allocate thrust;
we adapt the actuator parameters directly using on‑line wrench error.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------- Utility ---------------------------------

def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def wrap_angle(x, lo=-math.pi, hi=math.pi):
    rng = hi - lo
    return lo + (x - lo) % rng

# --------------------------- Vessel model -------------------------------
class Vessel3DOF:
    def __init__(self, m_x=500.0, m_y=600.0, I_z=800.0,
                 d_x=150.0, d_y=180.0, d_r=200.0):
        # Inertias (lumped)
        self.M = np.diag([m_x, m_y, I_z])
        self.D = np.diag([d_x, d_y, d_r])
        self.v = np.zeros(3)  # [u, v, r]
        self.eta = np.zeros(3)  # [x, y, psi]

    def step(self, tau, dt):
        # 
        # M vdot + D v = tau  -> vdot = M^{-1}(tau - D v)
        vdot = np.linalg.solve(self.M, tau - self.D @ self.v)
        self.v += vdot * dt
        # Kinematics (world frame update)
        u, v, r = self.v
        psi = self.eta[2]
        c, s = math.cos(psi), math.sin(psi)
        self.eta[0] += (c*u - s*v) * dt
        self.eta[1] += (s*u + c*v) * dt
        self.eta[2] = wrap_angle(psi + r * dt)
        return self.eta.copy(), self.v.copy()

# ------------------------ Thruster mapping ------------------------------
class TwoAzimuthThrusters:
    def __init__(self,
                 r1=np.array([ 1.2,  0.8]),  # [x,y] m from CG
                 r2=np.array([ 1.2, -0.8]),
                 f_bounds=(-1200.0, 1200.0),  # N (signed; map RPM→N externally if needed)
                 delta_bounds=(-math.radians(60), math.radians(60)),  # rad
                 max_df_dt=3000.0,   # N/s rate limit
                 max_dd_dt=math.radians(120)):  # rad/s rate limit
        self.r1 = r1; self.r2 = r2
        self.f_lo, self.f_hi = f_bounds
        self.d_lo, self.d_hi = delta_bounds
        self.max_df_dt = max_df_dt
        self.max_dd_dt = max_dd_dt
        # current parameters (forces and angles)
        self.f1 = 0.0; self.f2 = 0.0
        self.d1 = 0.0; self.d2 = 0.0

    def B_and_tau(self, f1, f2, d1, d2):
        # Force vectors in body frame
        c1, s1 = math.cos(d1), math.sin(d1)
        c2, s2 = math.cos(d2), math.sin(d2)
        F1 = np.array([f1*c1, f1*s1])
        F2 = np.array([f2*c2, f2*s2])
        # Wrench contributions
        X = F1[0] + F2[0]
        Y = F1[1] + F2[1]
        N = self.r1[0]*F1[1] - self.r1[1]*F1[0] + self.r2[0]*F2[1] - self.r2[1]*F2[0]
        tau = np.array([X,Y,N])
        # Jacobian wrt [f1,f2,d1,d2]
        dtaudf1 = np.array([c1, s1, self.r1[0]*s1 - self.r1[1]*c1])
        dtaudf2 = np.array([c2, s2, self.r2[0]*s2 - self.r2[1]*c2])
        dtaudd1 = np.array([-f1*s1,  f1*c1,  f1*(self.r1[0]*c1 + self.r1[1]*s1)])
        dtaudd2 = np.array([-f2*s2,  f2*c2,  f2*(self.r2[0]*c2 + self.r2[1]*s2)])
        J = np.column_stack([dtaudf1, dtaudf2, dtaudd1, dtaudd2])  # 3x4
        return tau, J

    def project_limits(self, f1, f2, d1, d2):
        f1 = sat(f1, self.f_lo, self.f_hi)
        f2 = sat(f2, self.f_lo, self.f_hi)
        d1 = sat(d1, self.d_lo, self.d_hi)
        d2 = sat(d2, self.d_lo, self.d_hi)
        return f1, f2, d1, d2

    def rate_limit(self, f1, f2, d1, d2, dt):
        f1 = sat(f1, self.f1 - self.max_df_dt*dt, self.f1 + self.max_df_dt*dt)
        f2 = sat(f2, self.f2 - self.max_df_dt*dt, self.f2 + self.max_df_dt*dt)
        d1 = sat(d1, self.d1 - self.max_dd_dt*dt, self.d1 + self.max_dd_dt*dt)
        d2 = sat(d2, self.d2 - self.max_dd_dt*dt, self.d2 + self.max_dd_dt*dt)
        return f1, f2, d1, d2

    def set_params(self, f1, f2, d1, d2):
        self.f1, self.f2, self.d1, self.d2 = f1, f2, d1, d2

# ------------------- Adaptive allocator (4 params) ----------------------
class AdaptiveAllocator4P:
    def __init__(self, plant: Vessel3DOF, az: TwoAzimuthThrusters,
                 Kp=np.diag([400.0, 400.0, 300.0]),
                 Kd=np.diag([120.0, 120.0, 100.0]),
                 alpha_f=1.5e-3, alpha_d=3.0e-3,  # gradient steps for force/angle
                 l2_reg=1e-6,
                 f_init=(0.0,0.0), d_init=(0.0,0.0)):
        self.plant = plant
        self.az = az
        self.Kp = Kp
        self.Kd = Kd
        self.alpha_f = alpha_f
        self.alpha_d = alpha_d
        self.l2 = l2_reg
        self.f1, self.f2 = f_init
        self.d1, self.d2 = d_init

    def desired_wrench(self, v_ref, v, vdot_ref=None):
        # PD in body velocities -> desired wrench
        e = v_ref - v
        edot = -v  # crude derivative (assuming v_ref_dot≈0); or use vdot_ref if provided
        tau_des = self.Kp @ e + self.Kd @ edot
        return tau_des, e

    def step(self, v_ref, dt):
        # 1) Outer loop -> tau_des
        tau_des, e = self.desired_wrench(v_ref, self.plant.v)
        # 2) Current wrench and Jacobian from thrusters
        tau, J = self.az.B_and_tau(self.f1, self.f2, self.d1, self.d2)
        # 3) Wrench error
        e_tau = tau - tau_des
        # 4) Gradient of cost 0.5*||e_tau||^2 + l2*||p||^2
        grad = J.T @ e_tau + self.l2 * np.array([self.f1, self.f2, self.d1, self.d2])
        # 5) Parameter update (separate steps for forces/angles)
        df1 = -self.alpha_f * grad[0]
        df2 = -self.alpha_f * grad[1]
        dd1 = -self.alpha_d * grad[2]
        dd2 = -self.alpha_d * grad[3]
        f1 = self.f1 + df1
        f2 = self.f2 + df2
        d1 = self.d1 + dd1
        d2 = self.d2 + dd2
        # 6) Constraints: rate + box
        f1, f2, d1, d2 = self.az.rate_limit(f1, f2, d1, d2, dt)
        f1, f2, d1, d2 = self.az.project_limits(f1, f2, d1, d2)
        # 7) Commit
        self.f1, self.f2, self.d1, self.d2 = f1, f2, d1, d2
        self.az.set_params(f1, f2, d1, d2)
        # 8) Apply to plant
        tau_cmd, _ = self.az.B_and_tau(f1, f2, d1, d2)
        eta, v = self.plant.step(tau_cmd, dt)
        return eta, v, tau_des, tau_cmd, (self.f1, self.f2, self.d1, self.d2), e, e_tau

# ------------------------------ Demo ------------------------------------
if __name__ == "__main__":
    np.random.seed(1)
    dt = 0.05
    T = 100.0
    N = int(T/dt)

    vessel = Vessel3DOF()
    az = TwoAzimuthThrusters()
    ctrl = AdaptiveAllocator4P(vessel, az)

    # Reference: lateral shift and small rotation while keeping low surge
    def v_ref_profile(t):
        # Smooth S-curve in sway; hold near zero surge; damp yaw
        v_ref = np.zeros(3)
        # trapezoid in sway
        vmax, a = 0.35, 0.15
        t1, t2, t3 = 3.0, 12.0, 18.0
        if t < t1:
            v_ref[1] = vmax * (t/t1)
        elif t < t2:
            v_ref[1] = vmax
        elif t < t3:
            v_ref[1] = vmax * (1 - (t - t2)/(t3 - t2))
        else:
            v_ref[1] = 0.0
        # tiny yaw to align
        v_ref[2] = 0.005 * math.sin(0.2*t)
        return v_ref

    logs = {k: [] for k in ["t","eta","v","tau_des","tau_cmd","p","e_v","e_tau"]}

    for k in range(N):
        t = k*dt
        v_ref = v_ref_profile(t)
        eta, v, tau_des, tau_cmd, p, e_v, e_tau = ctrl.step(v_ref, dt)
        # log
        logs["t"].append(t)
        logs["eta"].append(eta)
        logs["v"].append(v)
        logs["tau_des"].append(tau_des)
        logs["tau_cmd"].append(tau_cmd)
        logs["p"].append(p)
        logs["e_v"].append(e_v)
        logs["e_tau"].append(e_tau)

    # Convert logs
    t = np.array(logs["t"]) 
    eta = np.array(logs["eta"]) 
    v = np.array(logs["v"]) 
    tau_des = np.array(logs["tau_des"]) 
    tau_cmd = np.array(logs["tau_cmd"]) 
    p = np.array(logs["p"]) 
    e_v = np.array(logs["e_v"]) 
    e_tau = np.array(logs["e_tau"]) 

    # -------------------------- Plots -----------------------------------
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(2,2,1)
    plt.plot(t, v[:,0], label='u (surge)')
    plt.plot(t, v[:,1], label='v (sway)')
    plt.plot(t, v[:,2], label='r (yaw rate)')
    plt.title('Body velocities')
    plt.legend(); plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(t, tau_des[:,0], '--', label='X_des')
    plt.plot(t, tau_cmd[:,0], label='X_cmd')
    plt.plot(t, tau_des[:,1], '--', label='Y_des')
    plt.plot(t, tau_cmd[:,1], label='Y_cmd')
    plt.plot(t, tau_des[:,2], '--', label='N_des')
    plt.plot(t, tau_cmd[:,2], label='N_cmd')
    plt.title('Wrench tracking')
    plt.legend(); plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(t, p[:,0], label='f1 [N]')
    plt.plot(t, p[:,1], label='f2 [N]')
    plt.plot(t, np.degrees(p[:,2]), label='d1 [deg]')
    plt.plot(t, np.degrees(p[:,3]), label='d2 [deg]')
    plt.title('Adaptive parameters (forces & angles)')
    plt.legend(); plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(eta[:,0], eta[:,1], label='Track')
    plt.scatter([eta[0,0]],[eta[0,1]], label='Start')
    plt.title('Trajectory (world)')
    plt.axis('equal'); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()
