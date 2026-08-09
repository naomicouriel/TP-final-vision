#include <Servo.h>

Servo servo1;
Servo servo2;

const unsigned long DWELL_MS = 1000;  // dwell time at each position
const unsigned long tVuelta_MS = 1000;  // return travel time

const int BASE_SERVO_PLAT   = 90;
const int BASE_SERVO_ROT    = 90;

void setup() {
  servo1.attach(9);        // rotation servo pin
  servo2.attach(10);       // platform servo pin
  servo1.write(BASE_SERVO_ROT);  // move to HOME on startup
  servo2.write(BASE_SERVO_PLAT);  // move to HOME on startup
  
  delay(1000); 
}
// BIN QUADRANTS
// __________________
// |       |        |
// |  1    |   2    |
// |------ |------- |
// |  3    |   4    |
// |_______|________|

void loop() {
// MOVE TO QUADRANT 1
  servo1.write(BASE_SERVO_ROT - 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT + 40);
  delay(DWELL_MS);

  // RETURN HOME
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS + 500);

  // MOVE TO QUADRANT 2
  servo1.write(BASE_SERVO_ROT + 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT + 40);
  delay(DWELL_MS);

  // RETURN HOME
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

  // MOVE TO QUADRANT 3
  servo1.write(BASE_SERVO_ROT - 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT - 40);
  delay(DWELL_MS);
  

  // RETURN HOME
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

  // MOVE TO QUADRANT 4
  servo1.write(BASE_SERVO_ROT + 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT - 40);
  delay(DWELL_MS);

  // RETURN HOME
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

}