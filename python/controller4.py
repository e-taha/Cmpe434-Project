import time

class PIDController: 
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.start_time = None

    def proportional(self, error):
        return self.kp * error
    
    def integral_error(self, error):
        self.integral += error
        return self.ki * self.integral
    
    def derivative(self, error):
        dt = 0.002
        derivative = (error - self.prev_error)/dt
        self.prev_error = error
        # self.start_time = time.time()
        return self.kd * derivative
    
    def update(self, error):
        return self.proportional(error) + self.integral_error(error) + self.derivative(error)
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.start_time = None

        


   