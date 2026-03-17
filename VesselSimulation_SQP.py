
import socket, threading, time, math
from scipy.optimize import minimize
import traceback

# Custom atan2, sin, cos for compatibility
pi = math.pi
def atan2(y, x): return math.atan2(y, x)
def sin(x): return math.sin(x)
def cos(x): return math.cos(x)

# system port
SEND_IP = '000.000.00.0'
SEND_PORT = 32361
RECV_PORT = 32360

# machine limits
MAX_RPM = 690
MIN_RPM = -690
MAX_ANGLE = 25
MIN_ANGLE = -25
RR = 45.0    # rpm rising rate (70 rpm/s, 14 rpm/0.2s)
RF = 45.0     # rpm falling rate (45 rpm/s, 9 rpm/0.2s)
AC = 7.0   # angle changing rate (7 deg/s, 1.4 rpm/0.2s)
AZ2 = [[[-5.7,-1.0],[-5.7,1.0]], # -5.7
       [[],[]],
       [[],[]]]
PEROID = 0.34
DR = RR * PEROID
DF = RF * PEROID  
DA = AC * PEROID  
SCALE = 15

# tolerance
THRESH_FORWARD = 3
THRESH_LATERAL = 1
THRESH_HDG = 5  # heading

# target
TARGET_X = 25
TARGET_Y = 0
TARGET_HEADING = 0 # 360-10
# In notboke setting: TARGET_HEADING = 10 (difference between berthing heading and ship heading)

# PID 控制器
class PID:
    def __init__(self, Kp, Ki=0, Kd=0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

NEGADJ = 1.34
#dist_pid = PID(5, 0, 0.05)  # distance PID
forward_pid = PID(100, 0, 7100)  # forward PID
lateral_pid = PID(20, 0.0, 700)  # lateral PID
angle_pid = PID(100, 0.0, 3000)  # angle PID


def angle_diff(a, b):
    return (a - b + 180) % 360 - 180

def find_min(xyz,thrusters,showdetails=False):
    '''
    xyz: desired forces, Fx, Fy, Mz
    ths: thruster's place [x,y]
    tcn: thruster's constraint force, [], [fixed value], or [min,max], ignore values of i>1
    acn: thruster's constraint angel, same as above
    '''
    ### input
    start = time.time()
    ths = thrusters[0]
    tcn = thrusters[1]
    acn = thrusters[2]
    nt = len(ths)
    eps = 1e-9 # smoothing parameter

    ### Define the objective function
    def objective(x): # use square and add eps to prevent form minus x value that makes x**1.5 error  
        return sum([(abs(x[2*i])#*(3*(1-i)+i)
                     +eps)**1.5 for i in range(nt)])

    ### Define the equality constraint Fx, Fy, Mz
    def constraint_eqx(x): # Fx-sum(force_x) = 0
        return sum([x[2*i]*cos(x[2*i+1]) for i in range(nt)]) - xyz[0] 
    def constraint_eqy(x):
        return sum([x[2*i]*sin(x[2*i+1]) for i in range(nt)]) - xyz[1]
    def constraint_eqz(x): # Mz-sum(Fy*x-Fx*y) = 0
        return sum([x[2*i]*sin(x[2*i+1])*ths[i][0]-x[2*i]*cos(x[2*i+1])*ths[i][1] for i in range(nt)]) - xyz[2] 
    constraints = [{'type': 'eq', 'fun': constraint_eqx},
                   {'type': 'eq', 'fun': constraint_eqy},
                   {'type': 'eq', 'fun': constraint_eqz}]

    ### Define the other equality and inequality constraints
    def constraint_eq(index,value): # output == value
        return lambda x: x[index]-value
    def constraint_ineq_min(index,value): # output >= value
        return lambda x: x[index]-value
    def constraint_ineq_max(index,value): # value >= output
        return lambda x: value-x[index]
    for i in range(nt):
        if len(tcn[i]) == 1:
            constraints += [{'type': 'eq', 'fun': constraint_eq(i*2,tcn[i])}]
        elif len(tcn[i]) > 1:
            constraints += [{'type': 'ineq', 'fun': constraint_ineq_min(i*2,tcn[i][0])},
                            {'type': 'ineq', 'fun': constraint_ineq_max(i*2,tcn[i][1])}]
        if len(acn[i]) == 1:
            constraints += [{'type': 'eq', 'fun': constraint_eq(i*2+1,acn[i])}]
        elif len(acn[i]) > 1:
            constraints += [{'type': 'ineq', 'fun': constraint_ineq_min(i*2+1,acn[i][0])},
                            {'type': 'ineq', 'fun': constraint_ineq_max(i*2+1,acn[i][1])}]
    

    ### Initial guess
    x0 = [eps]*nt*2
    
    rfx = xyz[0]/nt # average for each thruster
    rfy = xyz[1]/nt
    for i in range(nt):
        # thrusts
        xi,yi = ths[i]
        if xi or yi:
            rfx -= yi*xyz[2]/(nt*(xi**2+yi**2))
            rfy += xi*xyz[2]/(nt*(xi**2+yi**2))
        x0[2*i] = (rfx**2+rfy**2)**0.5+eps
        # angles
        if len(acn[i]) == 1:
            x0[2*i+1] = acn[i][0]
        elif rfx or rfy:
            ang = atan2(rfy,rfx)
            if len(acn[i]) > 1:
                x0[2*i+1] = min(max(ang,acn[i][0]),acn[i][1])
            else:
                x0[2*i+1] = ang
    

    ### Solve the optimization problem
    solution = minimize(objective,
                        x0, 
                        constraints=constraints, 
                        method='SLSQP',
                        tol=1e-6, 
                        options={'disp': showdetails, 'maxiter': 1000})

    ### Extract results
    x_opt = solution.x
    for i in range(nt):
        x_opt[2*i+1] = x_opt[2*i+1]%(2*pi)*180/pi
    f_opt = solution.fun

    ### Print the results
    takentime = time.time() - start
    if showdetails:
        print(f'Execution Time: {takentime*1000:.3f} ms')
        print("Optimal Solution:")
        print(x_opt)
        print(f"Optimal Objective Value = {f_opt:.3f}")
    return x_opt

def transfer(rpm, angle):
    if angle < 90:
        angle = -angle
    elif angle < 270:
        angle = 180 - angle
        rpm = -rpm
    else:
        angle = 360 - angle
    if rpm < 0:
        rpm = rpm * NEGADJ
    return [rpm, angle]

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_command(l_rpm, l_ang, r_rpm, r_ang):  
    l_rpm = int(min(max(l_rpm, MIN_RPM), MAX_RPM))
    r_rpm = int(min(max(r_rpm, MIN_RPM), MAX_RPM))
    l_ang = int(min(max(l_ang, MIN_ANGLE), MAX_ANGLE))
    r_ang = int(min(max(r_ang, MIN_ANGLE), MAX_ANGLE))
    cmd = f"$OBCMD,{l_rpm},{l_ang},{r_rpm},{r_ang}*3A"
    send_sock.sendto(cmd.encode(), (SEND_IP, SEND_PORT))
    print("[Send]", cmd)

def control_loop():
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(('', RECV_PORT))
    t0 = -PEROID  # Initialize t0 to a negative value to avoid immediate dt calculation
    forward_err, lateral_err, yaw_err = TARGET_Y, TARGET_X, angle_diff(TARGET_HEADING, 0)
    x0, y0, a0 = 0, 0, 0
    vx0, vy0, va0 = 0, 0, 0
    vx, vy, va = 0, 0, 0
    ax, ay, aa = 0, 0, 0
    while True or (abs(forward_err)>THRESH_FORWARD or abs(lateral_err)>THRESH_LATERAL or abs(yaw_err)>THRESH_HDG
           or abs(vx) > 0.01 or abs(vy) > 0.01 or abs(va) > 0.01 or abs(ax) > 0.01 or abs(ay) > 0.01 or abs(aa) > 0.01):
        data, _ = recv_sock.recvfrom(1024)
        msg = data.decode().strip()

        if not msg.startswith("$BSPOI"): continue

        try:
            _, t, x, y, hdg, lrn, lan, rrn, ran = msg.split(",")
            t, x, y, hdg, lrn, lan, rrn, ran = float(t), float(x), float(y), float(hdg), float(lrn), float(lan), float(rrn), float(ran[:-3])
            dt = t - t0
            if dt < PEROID:
                #print(f"dt={dt:.3f} < {PEROID}, skipping this loop")
                continue
            t0 = t
            vx = (x - x0) / dt
            vy = (y - y0) / dt
            va = angle_diff(hdg, a0) / dt
            ax = (vx - vx0) / dt
            ay = (vy - vy0) / dt
            aa = (va - va0) / dt
            x0, y0, a0 = x, y, hdg
            vx0, vy0, va0 = vx, vy, va
            
            # global coordinate
            '''
            y (0 deg)
            ^ 
            L-→ x (90 deg) 
            '''
            dx = TARGET_X - x
            dy = TARGET_Y - y
            dist_total = math.hypot(dx, dy) # distance d 
            yaw_err = angle_diff(TARGET_HEADING, hdg) # theta, parallel between berthing heading and ship heading, more important in berthing   

            # ship coordinate
            # system's thruster coordinate:
            # +deg | -deg
            #    \ | / 
            '''
            optimization vessel model (thruster) coordinate :
            x (0 deg)
            ^ 
            L-→ y (90 deg) 
            '''
            rad = math.radians(hdg)
            forward_err = dx * math.sin(rad) + dy * math.cos(rad) 
            lateral_err = dx * math.cos(rad) - dy * math.sin(rad) 
            phi = math.atan2(lateral_err, forward_err) # lateral/forward
            # hdg_err = angle_diff(math.pi/2, phi) # head points at the goal, more important in cruising

            print(f"[Receive] X={x:.1f} Y={y:.1f} H={hdg:.1f} → dist={dist_total:.1f} fwd={forward_err:.1f} lat={lateral_err:.1f} phi={phi:.1f}(rad) theta={yaw_err:.1f}(deg) ")
            #force = dist_pid.update(dist_total,dt)
            forward_force = forward_pid.update(forward_err, dt)
            lateral_force = lateral_pid.update(lateral_err, dt)
            torq = angle_pid.update(yaw_err,dt) # clockwise positive
            xyz = [forward_force, lateral_force, torq]
            print("Desired Forces:", xyz)
            #xyz = [force * cos(phi), force * sin(phi), torq]  # desired forces, Fx, Fy, Mz
            lr, la, rr, ra = find_min(xyz, AZ2)
            lr, la = transfer(lr, la)
            rr, ra = transfer(rr, ra)
            lrup, lrlow, rrup, rrlow = min(lrn+DR,MAX_RPM), max(lrn-DF,MIN_RPM), min(rrn+DR,MAX_RPM), max(rrn-DF,MIN_RPM)
            print('origin',lr,rr,lrup,lrlow,rrup,rrlow)
            if lrup > 0 and lr > lrup:
                rr *= lrup / lr
                lr = lrup
                print('leftup',lr,rr)
            if rrup > 0 and rr > rrup:
                lr *= rrup / rr
                rr = rrup
                print('rightup',lr,rr)
            if lrlow < 0 and lr < lrlow:
                rr *= lrlow / lr
                lr = lrlow
                print('leftlow',lr,rr)
            if rrlow < 0 and rr < rrlow:
                lr *= rrlow / rr
                rr = rrlow
                print('rightlow',lr,rr)
            lr = (min(max(lr, lrn-DF), lrn+DR))//15*15
            la = (min(max(la, lan-DA), lan+DA))
            rr = (min(max(rr, rrn-DF), rrn+DR))//15*15        
            ra = (min(max(ra, ran-DA), ran+DA))

            send_command(lr, la, rr, ra)

        except Exception as e:
            print("[ERROR]", e)
            print(t, t0, dt)
            traceback.print_exc()
    
    # Stop the thrusters when the target is reached or conditions are met
    send_command(0, 0, 0, 0)
    print("Target reached or conditions met, stopping control loop.")
    print("Final State: Time =", t0,"Forward Error =", forward_err, "Lateral Error =", lateral_err, "Yaw Error =", yaw_err)

# start
if __name__ == '__main__':
    threading.Thread(target=control_loop, daemon=True).start()
    input("Start\n")
