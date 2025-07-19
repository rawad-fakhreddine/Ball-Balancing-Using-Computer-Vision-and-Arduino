#include <AccelStepper.h>

// Stepper motor pins
#define STEP_X 3
#define DIR_X 4
#define STEP_Y 6
#define DIR_Y 7
#define DRIVER 1   // A4988/DRV8825

// Create stepper objects
AccelStepper stepperX(DRIVER, STEP_X, DIR_X);
AccelStepper stepperY(DRIVER, STEP_Y, DIR_Y);

// Constants
const float MICROSTEPS     = 1;
const float STEPS_PER_REV  = 200.0;       // 1.8° per step
const float STEPS_PER_DEG  = (STEPS_PER_REV * MICROSTEPS) / 360.0; 
const float MAX_TILT       = 25.0;        // ±25° limit

String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(9600);
  inputString.reserve(32);

  // Motor settings
  stepperX.setMaxSpeed(600);
  stepperX.setAcceleration(200);
  stepperY.setMaxSpeed(600);
  stepperY.setAcceleration(200);
  
  stepperX.setCurrentPosition(0);
  stepperY.setCurrentPosition(0);

  Serial.println("READY");
}

void loop() {
  // Run motors continuously
  stepperX.run();
  stepperY.run();

  // Process complete serial message
  if (stringComplete) {
    stringComplete = false;
    float angleX = 0, angleY = 0;

    int xIndex = inputString.indexOf('X');
    int commaIndex = inputString.indexOf(',');
    int yIndex = inputString.indexOf('Y');

    if (xIndex == 0 && commaIndex > xIndex && yIndex > commaIndex) {
      String xStr = inputString.substring(xIndex + 1, commaIndex);
      String yStr = inputString.substring(yIndex + 1);

      angleX = xStr.toFloat();
      angleY = yStr.toFloat();

      angleX = constrain(angleX, -MAX_TILT, MAX_TILT);
      angleY = constrain(angleY, -MAX_TILT, MAX_TILT);

      int stepsX = angleX * STEPS_PER_DEG;
      int stepsY = angleY * STEPS_PER_DEG;

      stepperX.moveTo(stepsX);
      stepperY.moveTo(stepsY);

      // Optional debug print
      Serial.print("Moving to: X=");
      Serial.print(angleX);
      Serial.print("°, Y=");
      Serial.print(angleY);
      Serial.println("°");
    }

    // // Send acknowledgment after processing
    // Serial.println("ACK");

    inputString = ""; // reset
  }
}

// Collect incoming serial characters
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
