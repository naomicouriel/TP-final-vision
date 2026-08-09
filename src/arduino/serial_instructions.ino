
// REFERENCE SKETCH FOR THE ARDUINO
// THE ACTUAL SKETCH IS FLASHED DIRECTLY FROM THE ARDUINO IDE

#include <Servo.h>

Servo servo1;
Servo servo2;

const unsigned long DWELL_MS = 2000;  // dwell time at each position
const unsigned long tVuelta_MS = 200;  // return travel time
const unsigned long SERIAL_TIMEOUT = 5000;  // serial read timeout (ms)

const int BASE_SERVO_PLAT   = 90;
const int BASE_SERVO_ROT    = 88;


void setup() {
  Serial.begin(9600);  // Initialise serial communication at 9600 baud
  servo1.attach(9);        // servo pin
  servo2.attach(10);
  servo1.write(BASE_SERVO_ROT);  // move to HOME on startup
  servo2.write(BASE_SERVO_PLAT);  // move to HOME on startup
  
  Serial.println("Arduino ready. Waiting for instructions (0-3)...");
  delay(1000); 
}

// BIN QUADRANTS (mapped to the model classes)
// __________________
// |       |        |
// |  0    |   1    |  Class 0 (cardboard_paper) -> Quadrant 0
// |------ |------- |  Class 1 (ecoglasses)      -> Quadrant 1
// |  2    |   3    |  Class 2 (metal_plastic)   -> Quadrant 2
// |_______|________|  Class 3 (trash)           -> Quadrant 3

void moveToQuadrant(int quadrant) {
  switch(quadrant) {
    case 0:  // Top-left quadrant
      servo1.write(BASE_SERVO_ROT - 45);
      delay(DWELL_MS);
      servo2.write(BASE_SERVO_PLAT + 40);
      delay(DWELL_MS);
      break;
      
    case 1:  // Top-right quadrant
      servo1.write(BASE_SERVO_ROT + 45);
      delay(DWELL_MS);
      servo2.write(BASE_SERVO_PLAT + 40);
      delay(DWELL_MS);
      break;
      
    case 2:  // Bottom-left quadrant
      servo1.write(BASE_SERVO_ROT - 45);
      delay(DWELL_MS);
      servo2.write(BASE_SERVO_PLAT - 40);
      delay(DWELL_MS);
      break;
      
    case 3:  // Bottom-right quadrant
      servo1.write(BASE_SERVO_ROT + 45);
      delay(DWELL_MS);
      servo2.write(BASE_SERVO_PLAT - 40);
      delay(DWELL_MS);
      break;
  }
  
  // Return to home position
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS + 500);
}

void loop() {
  // Wait for a serial instruction (0-3)
  if (Serial.available() > 0) {
    int instruction = Serial.read();
    
    // Convert the ASCII character to a number (0-3)
    if (instruction >= '0' && instruction <= '3') {
      int quadrant = instruction - '0';
      
      Serial.print("Moving to quadrant: ");
      Serial.println(quadrant);
      
      moveToQuadrant(quadrant);
      
      Serial.println("Done.");
    } else {
      // Ignore invalid characters
      Serial.print("Invalid instruction: ");
      Serial.println(instruction);
    }
  }
}

