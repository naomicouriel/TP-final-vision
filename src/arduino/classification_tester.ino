#include <Servo.h>

Servo servo1;
Servo servo2;

const unsigned long DWELL_MS = 1000;  // tiempo en cada posición
const unsigned long tVuelta_MS = 1000;  // tiempo de vuelta

const int BASE_SERVO_PLAT   = 90;
const int BASE_SERVO_ROT    = 90;

void setup() {
  servo1.attach(9);        // pin del servo de rotación
  servo2.attach(10);       // pin del servo de plataforma
  servo1.write(BASE_SERVO_ROT);  // ir a HOME al inicio
  servo2.write(BASE_SERVO_PLAT);  // ir a HOME al inicio
  
  delay(1000); 
}
// CUADRANTES DEL TACHO
// __________________
// |       |        |
// |  1    |   2    |
// |------ |------- |
// |  3    |   4    |
// |_______|________|

void loop() {
// MOVETE AL 1 CUADRANTE
  servo1.write(BASE_SERVO_ROT - 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT + 40);
  delay(DWELL_MS);

  // VOLVE A CASA
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS + 500);

  // MOVETE AL 2 CUADRANTE
  servo1.write(BASE_SERVO_ROT + 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT + 40);
  delay(DWELL_MS);

  // VOLVE A CASA
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

  // MOVETE AL 3 CUADRANTE
  servo1.write(BASE_SERVO_ROT - 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT - 40);
  delay(DWELL_MS);
  

  // VOLVE A CASA
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

  // MOVETE AL 4 CUADRANTE
  servo1.write(BASE_SERVO_ROT + 45);
  delay(DWELL_MS);
  servo2.write(BASE_SERVO_PLAT - 40);
  delay(DWELL_MS);

  // VOLVE A CASA
  servo2.write(BASE_SERVO_PLAT);
  delay(tVuelta_MS);
  servo1.write(BASE_SERVO_ROT);
  delay(DWELL_MS+ 500);

}