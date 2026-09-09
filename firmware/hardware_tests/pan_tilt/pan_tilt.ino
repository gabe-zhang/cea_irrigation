/*
 * Pan & Tilt Calibration: pi50
 * For absolute angle:
 *   Home: p 55, t 30
 *   Range:
 *     p +- 65 (0 - 130)
 *     t 0 - 60
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN  = 9;
const int TILT_PIN = 10;

// Calibrated home positions
const int PAN_HOME  = 55;
const int TILT_HOME = 30;

void setup() {
  Serial.begin(9600);
  
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  // Home both servos at calibrated positions on startup
  panServo.write(PAN_HOME);
  tiltServo.write(TILT_HOME);

  Serial.println("=== Pan & Tilt Controller Ready ===");
  Serial.println("Default Home: Pan = 55 deg, Tilt = 30 deg");
  Serial.println("Commands:");
  Serial.println("  p <angle>  -> Rotate Pan  (e.g., p 55, p 0, p 130)");
  Serial.println("  t <angle>  -> Move Tilt   (e.g., t 30, t 0, t 60)");
  Serial.println("  h          -> Home        (Pan 55, Tilt 30)");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) {
      return;
    }

    // Command 'h' or legacy 'c' to quickly return to calibrated home
    if (cmd.equalsIgnoreCase("h") || cmd.equalsIgnoreCase("c")) {
      panServo.write(PAN_HOME);
      tiltServo.write(TILT_HOME);
      Serial.println("ACK: Homed to Pan 55 deg, Tilt 30 deg");
      return;
    }

    if (cmd.length() < 2) {
      return;
    }

    char action = cmd.charAt(0);
    int angle = cmd.substring(1).toInt();

    if (action == 'p' || action == 'P') {
      angle = constrain(angle, 0, 130);
      panServo.write(angle);
      Serial.print("ACK: Pan rotated to ");
      Serial.print(angle);
      Serial.println(" deg");
    } 
    else if (action == 't' || action == 'T') {
      angle = constrain(angle, 0, 60);
      tiltServo.write(angle);
      Serial.print("ACK: Tilt moved to ");
      Serial.print(angle);
      Serial.println(" deg");
    } 
    else {
      Serial.print("ERR: Unknown command '");
      Serial.print(cmd);
      Serial.println("'. Use 'p <0-130>', 't <0-60>', 'h', or 'c'");
    }
  }
}
