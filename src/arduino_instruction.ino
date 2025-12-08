// CODIGO DE REFERENCIA PARA EL ARDUINO
// EL CODIGO REAL SE CORRE DIRECTAMENTE DESDE EL ARDUINO IDE

// #include <Servo.h>

// Servo servo1;
// Servo servo2;

// const unsigned long DWELL_MS = 2000;  // tiempo en cada posición
// const unsigned long tVuelta_MS = 200;  // tiempo de vuelta
// const unsigned long SERIAL_TIMEOUT = 5000;  // timeout para lectura serial (ms)

// const int BASE_SERVO_PLAT   = 90;
// const int BASE_SERVO_ROT    = 88;


// void setup() {
//   Serial.begin(9600);  // Inicializar comunicación serial a 9600 baud
//   servo1.attach(9);        // pin del servo
//   servo2.attach(10);
//   servo1.write(BASE_SERVO_ROT);  // ir a HOME al inicio
//   servo2.write(BASE_SERVO_PLAT);  // ir a HOME al inicio
  
//   Serial.println("Arduino listo. Esperando instrucciones (0-3)...");
//   delay(1000); 
// }

// // CUADRANTES DEL TACHO (Mapeado a clases del modelo)
// // __________________
// // |       |        |
// // |  0    |   1    |  Clase 0 (cardboard_paper) -> Cuadrante 0
// // |------ |------- |  Clase 1 (ecoglasses)      -> Cuadrante 1
// // |  2    |   3    |  Clase 2 (metal_plastic)   -> Cuadrante 2
// // |_______|________|  Clase 3 (trash)           -> Cuadrante 3

// void moveToQuadrant(int quadrant) {
//   switch(quadrant) {
//     case 0:  // Cuadrante superior izquierdo
//       servo1.write(BASE_SERVO_ROT - 45);
//       delay(DWELL_MS);
//       servo2.write(BASE_SERVO_PLAT + 40);
//       delay(DWELL_MS);
//       break;
      
//     case 1:  // Cuadrante superior derecho
//       servo1.write(BASE_SERVO_ROT + 45);
//       delay(DWELL_MS);
//       servo2.write(BASE_SERVO_PLAT + 40);
//       delay(DWELL_MS);
//       break;
      
//     case 2:  // Cuadrante inferior izquierdo
//       servo1.write(BASE_SERVO_ROT - 45);
//       delay(DWELL_MS);
//       servo2.write(BASE_SERVO_PLAT - 40);
//       delay(DWELL_MS);
//       break;
      
//     case 3:  // Cuadrante inferior derecho
//       servo1.write(BASE_SERVO_ROT + 45);
//       delay(DWELL_MS);
//       servo2.write(BASE_SERVO_PLAT - 40);
//       delay(DWELL_MS);
//       break;
//   }
  
//   // Volver a casa
//   servo2.write(BASE_SERVO_PLAT);
//   delay(tVuelta_MS);
//   servo1.write(BASE_SERVO_ROT);
//   delay(DWELL_MS + 500);
// }

// void loop() {
//   // Esperar instrucción por serial (0-3)
//   if (Serial.available() > 0) {
//     int instruction = Serial.read();
    
//     // Convertir carácter ASCII a número (0-3)
//     if (instruction >= '0' && instruction <= '3') {
//       int quadrant = instruction - '0';
      
//       Serial.print("Moviendo a cuadrante: ");
//       Serial.println(quadrant);
      
//       moveToQuadrant(quadrant);
      
//       Serial.println("Listo.");
//     } else {
//       // Ignorar caracteres no válidos
//       Serial.print("Instrucción inválida: ");
//       Serial.println(instruction);
//     }
//   }
// }
