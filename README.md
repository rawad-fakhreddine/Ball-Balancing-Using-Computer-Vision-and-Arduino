# Ball-Balancing-Using-Computer-Vision-and-Arduino
This project explores the use of computer vision, reinforcement learning, and motor control to dynamically stabilize a ball on a tilting platform

# Explanation of the Project # 
A computer vision-based control system that autonomously balances a tennis ball at the center of a tilting platform using Reinforcement Learning (RL), YOLOv8, and Arduino-controlled stepper motors.

# Project Overview: #
This project simulates and implements a ball-on-platform stabilizing system, where the goal is to keep a tennis ball centered using real-time detection and control. Unlike traditional PID controllers, this system uses a trained RL agent that makes decisions based on visual feedback from a webcam.

# Objectives: #
	Detect and track the tennis ball using YOLOv8 and OpenCV

	Calculate real-world position, velocity, and acceleration from webcam input

	Train a PPO reinforcement learning agent to stabilize the ball

	Send motor commands to an Arduino using serial communication

	Control platform tilt with stepper motors via mechanical linkages

# System Architecture: #
1. Computer Vision (YOLOv8)
	YOLOv8 model detects the ball's position in webcam frames

	Coordinates are converted from pixels to centimeters

2. State Estimation
	Ball velocity and acceleration are estimated using time-differentiation

	Data is filtered using a smoothing algorithm and a deadband filter

3. Reinforcement Learning Agent
	PPO agent trained using Stable-Baselines3 in a custom Gymnasium environment

	Inputs: [x, y, vx, vy]

	Outputs: Normalized tilt angles in X and Y axes

	Curriculum learning is used to gradually increase difficulty during training

4. Arduino & Motor Control
	Arduino receives angles via serial port

	Uses AccelStepper to drive 2 NEMA 17 stepper motors

	Stepper motors tilt the platform via mechanical linkages

# Hardware Used: #
	Arduino UNO

	2× NEMA 17 Stepper Motors

	A4988 Motor Drivers

	Webcam (top-down view)

	3D-Printed Mechanical Linkage System

	Laptop running Python for YOLOv8 + RL
