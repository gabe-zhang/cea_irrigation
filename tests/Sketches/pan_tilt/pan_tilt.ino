/*
 * Pan & Tilt Calibration: pi50
 * For absolute angle:
 *   Centered: p 65, t 60
 *   Range:
 *     p +- 65 (0 - 130)
 *     t 0 - 90
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN  = A4;
const int TILT_PIN = A5;

// Calibrated center positions
const int PAN_CENTER  = 65;
const int TILT_CENTER = 60;

void setup() {
  Serial.begin(9600);
  
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  // Center both servos at calibrated positions on startup
  panServo.write(PAN_CENTER);
  tiltServo.write(TILT_CENTER);

  Serial.println("=== Pan & Tilt Controller Ready ===");
  Serial.println("Default Center: Pan = 65 deg, Tilt = 60 deg");
  Serial.println("Commands:");
  Serial.println("  p <angle>  -> Rotate Pan  (e.g., p 65, p 0, p 180)");
  Serial.println("  t <angle>  -> Move Tilt   (e.g., t 60, t 45, t 135)");
  Serial.println("  c          -> Re-center   (Pan 65, Tilt 60)");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) {
      return;
    }

    // Command 'c' to quickly return to calibrated center
    if (cmd.equalsIgnoreCase("c")) {
      panServo.write(PAN_CENTER);
      tiltServo.write(TILT_CENTER);
      Serial.println("ACK: Re-centered to Pan 65 deg, Tilt 60 deg");
      return;
    }

    if (cmd.length() < 2) {
      return;
    }

    char action = cmd.charAt(0);
    int angle = cmd.substring(1).toInt();
    angle = constrain(angle, 0, 180);

    if (action == 'p' || action == 'P') {
      panServo.write(angle);
      Serial.print("ACK: Pan rotated to ");
      Serial.print(angle);
      Serial.println(" deg");
    } 
    else if (action == 't' || action == 'T') {
      tiltServo.write(angle);
      Serial.print("ACK: Tilt moved to ");
      Serial.print(angle);
      Serial.println(" deg");
    } 
    else {
      Serial.print("ERR: Unknown command '");
      Serial.print(cmd);
      Serial.println("'. Use 'p <0-180>', 't <0-180>', or 'c'");
    }
  }
}
