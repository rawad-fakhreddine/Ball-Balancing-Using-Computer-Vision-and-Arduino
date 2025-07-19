import cv2
import time
import os
import math
import serial
import numpy as np
from ultralytics import YOLO
from scipy.optimize import root_scalar
from stable_baselines3 import PPO


# ——— Configuration ———
rl_model = PPO.load("theta2_agent_v2")
model_path = r'C:\Users\Rawad\Desktop\Arduino\test_2\training\models\best.pt'

# —— 1) Initialise All The Data
prev_time = time.time()
prev_pos = (0, 0)
prev_velocity = (0, 0)
prev_smoothed_velocity = (0, 0)
prev_smoothed_acceleration = (0, 0)
alpha = 0.5                        # smoothing factor
pixel_to_cm = round(14.3/640,3)    # from pixel to cm 

# —— 2) For Communication With Arduino
SERIAL_PORT     = "COM4"
BAUD            = 9600

# —— 3) Stepper Motor Parameters
microsteps = 1                                          # 1/16‑step on A4988 (match MS1‑MS3 jumpers)
steps_per_rev = 200                                     # 1.8° motor
steps_per_degree = (steps_per_rev*microsteps)/360.0     # 8.888 step per 1 degree if microstep = 16
max_angle = 25                                           # limit your tilt to ±25°

# —— 4) PID Parameters
Kp = 2.0    # proportional gain
Ki = 1      # integral gain
Kd = 0.5    # derivative gain
g = 9.81                           # gravity in m/s² 
pos_deadband_m = 0.005             # 5 mm deadband around center


# ——— Load Model ———
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()
model = YOLO(model_path)


# ——— Open Camera ———
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()


# ——— Get image dimensions ———
ret, frame = cap.read()
if not ret:
    print("Failed to read initial frame.")
    cap.release()
    exit()
frame_height, frame_width = frame.shape[:2]
reference_x, reference_y = frame_width // 2, frame_height // 2


# ——— Open Serial ———
ser = serial.Serial(SERIAL_PORT, BAUD)
time.sleep(2)  # Wait for Arduino to reset 


# ——— Linkage Function (θ₄→θ₂) ———
def theta4_to_theta2(theta4_deg: float) -> float:
    theta4_rad = math.radians(theta4_deg)

    def f(theta2_rad):
        return 4.05 + 3.4948 * (math.cos(theta2_rad) - math.cos(theta4_rad)) - 2 * math.cos(theta2_rad - theta4_rad)

    # Solve between 0 and 180 degrees (in radians)
    sol = root_scalar(f, bracket=[0, math.pi], method='brentq')

    if sol.converged:
        return math.degrees(sol.root)
    else:
        raise RuntimeError("Root finding did not converge")


# ——— Start the Loop ———
while True:
    
    #  ——— Start Detection And Finding Position Of The Ball  ———
    ret, frame = cap.read()
    if not ret:
        break
  
    results = model.predict(source=frame, imgsz=640, conf=0.5, device='cpu', verbose=False)
    current_time = time.time()
    dt = current_time - prev_time if prev_time else 1e-3

    if results and results[0].boxes:
        box = results[0].boxes[0]
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, xyxy)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # —— 1)  Draw reference center 
        cv2.circle(frame, (reference_x, reference_y), 5, (0, 0, 255), -1)   
        
        # —— 2) Draw detection box and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # —— 3) Relative to center of frame
        pixel_x = center_x - reference_x
        pixel_y = center_y - reference_y
        
        rel_x = round(pixel_x*pixel_to_cm,3)
        rel_y = round(pixel_y*pixel_to_cm,3)
       
        noise_threshold = 0.25  # Noise threshold in pixels
        if abs(rel_x - prev_pos[0]) < noise_threshold and abs(rel_y - prev_pos[1]) < noise_threshold:
            rel_x = prev_pos[0]
            rel_y = prev_pos[1]
            
        pos = (rel_x, rel_y)

        # —— 4) Raw velocity
        velocity = ((rel_x - prev_pos[0]) / dt, (rel_y - prev_pos[1]) / dt)

        # —— 5) Smoothed velocity
        smoothed_velocity = (
            alpha * velocity[0] + (1 - alpha) * prev_smoothed_velocity[0],
            alpha * velocity[1] + (1 - alpha) * prev_smoothed_velocity[1]
        )

        # —— 6) Raw acceleration
        acceleration = (
            (velocity[0] - prev_velocity[0]) / dt,
            (velocity[1] - prev_velocity[1]) / dt
        )

        # —— 7) Smoothed acceleration
        smoothed_acceleration = (
            alpha * acceleration[0] + (1 - alpha) * prev_smoothed_acceleration[0],
            alpha * acceleration[1] + (1 - alpha) * prev_smoothed_acceleration[1]
        )
        
        # ——— Start PID Control ———
        
        # —— 1) initialize integrators & errors first time
        if 'int_x' not in locals():
            int_x = int_y = 0.0
            prev_err_x = prev_err_y = 0.0

        # —— 2) convert from cm to meter
        pos_m = (pos[0] * 0.01,  
                pos[1] * 0.01)

        vel_mps = (smoothed_velocity[0] * 0.01,
                smoothed_velocity[1] * 0.01)

        # # —— 3) compute errors (want ball→0)
        # err_x = -pos_m[0]
        # err_y = -pos_m[1]

        # # —— 4) deadband around origin
        # if abs(err_x) < pos_deadband_m:
        #     err_x = 0.0
        # if abs(err_y) < pos_deadband_m:
        #     err_y = 0.0

        # # —— 5) integrate 
        # int_x += err_x * dt
        # int_y += err_y * dt

        # # —— 6) PID Control
        # pid_x = (Kp * err_x) + (Ki * int_x) - (Kd * vel_mps[0]) 
        # pid_y = (Kp * err_y) + (Ki * int_y) - (Kd * vel_mps[1])

        # # —— 7) map desired acceleration → table tilt (θ₄, rad)
        # arg_x = max(-1.0, min(1.0, pid_x / g))
        # arg_y = max(-1.0, min(1.0, pid_y / g))
        
        # theta4_x = math.degrees(math.asin(arg_x))          
        # theta4_y = math.degrees(math.asin(arg_y))         

        # # —— 8) safety clamp (±MAX_TILT)
        # theta4_x = max(-max_angle, min(max_angle, theta4_x))
        # theta4_y = max(-max_angle, min(max_angle, theta4_y))
        
        # # —— 9) linkage conversion: table-tilt → motor angle (θ₂, deg)
        # theta2_x = theta4_to_theta2(theta4_x) - 100
        # theta2_y = theta4_to_theta2(theta4_y) - 100 

        ###### Reinforcment Learning
        # Construct observation: [x_position, y_position, x_velocity, y_velocity]
        obs = np.array([pos_m[0], pos_m[1], vel_mps[0], vel_mps[1]], dtype=np.float32)

        # Predict motor angles from RL model
        action, _ = rl_model.predict(obs, deterministic=True)

        theta4_x = float(action[0]) * max_angle
        theta4_y = float(action[1]) * max_angle

        theta4_x = max(-25, min(25, theta4_x))
        theta4_y = max(-25, min(25, theta4_y))
        
        if pos_m[0] > 0 and pos_m[1] > 0:
            if theta4_x < 0:
                theta4_x = -theta4_x
            if theta4_y < 0:
                theta4_y = -theta4_y

        elif pos_m[0] > 0 and pos_m[1] < 0:
            if theta4_x < 0:
                theta4_x = -theta4_x
            if theta4_y > 0:
                theta4_y = -theta4_y

        elif pos_m[0] < 0 and pos_m[1] > 0:
            if theta4_x > 0:
                theta4_x = -theta4_x
            if theta4_y < 0:
                theta4_y = -theta4_y

        elif pos_m[0] < 0 and pos_m[1] < 0:
            if theta4_x > 0:
                theta4_x = -theta4_x
            if theta4_y > 0:
                theta4_y = -theta4_y
        
        theta2_x = theta4_to_theta2(theta4_x)-110
        theta2_y = theta4_to_theta2(theta4_y)-110
        
        if pos_m[0] > 0 and pos_m[1] > 0:
            if theta2_x < 0:
                theta2_x = -theta2_x
            if theta2_y < 0:
                theta2_y = -theta2_y

        elif pos_m[0] > 0 and pos_m[1] < 0:
            if theta2_x < 0:
                theta2_x = -theta2_x
            if theta2_y > 0:
                theta2_y = -theta2_y

        elif pos_m[0] < 0 and pos_m[1] > 0:
            if theta2_x > 0:
                theta2_x = -theta2_x
            if theta2_y < 0:
                theta2_y = -theta2_y

        elif pos_m[0] < 0 and pos_m[1] < 0:
            if theta2_x > 0:
                theta2_x = -theta2_x
            if theta2_y > 0:
                theta2_y = -theta2_y
        
        # —— 10) Format as string 
        data_str = f"X{theta2_x:.2f},Y{theta2_y:.2f}\n"
        ser.write(data_str.encode())  # Send as bytes
        time.sleep(0.05)        # Sleep to send at 40Hz
        print(f"Moving to: X={theta2_x:.2f}°, Y={theta2_y:.2f}°")


        # —— 11) Show values on frame
        cv2.putText(frame, f"Rel Pos: ({rel_x}, {rel_y})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Vel: ({smoothed_velocity[0]:.2f}, {smoothed_velocity[1]:.2f})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Acc: ({smoothed_acceleration[0]:.2f}, {smoothed_acceleration[1]:.2f})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"Tilt X: { theta4_x:.2f} deg", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(frame, f"Tilt Y: { theta4_y:.2f} deg", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Theta X: { theta2_x:.2f} deg", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(frame, f"Theta Y: { theta2_y:.2f} deg", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


        # —— 12) Update previous values
        prev_pos = pos
        prev_velocity = velocity
        prev_smoothed_velocity = smoothed_velocity
        prev_smoothed_acceleration = smoothed_acceleration
        prev_time = current_time
        # prev_err_x = err_x
        # prev_err_y = err_y

    # Show frame
    cv2.imshow("Tennis Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ——— Close Serial And Destroy Window ———
ser.close()
cap.release()
cv2.destroyAllWindows()




