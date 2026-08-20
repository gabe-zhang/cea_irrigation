/*
 * 5-Relay & Water Pump Controller (5-Digit Bitmask Protocol)
 *
 * Pinout:
 *   - Relay 1 (Pump 1): Digital Pin 2
 *   - Relay 2 (Pump 2): Digital Pin 3
 *   - Relay 3 (Pump 3): Digital Pin 4
 *   - Relay 4 (Pump 4): Digital Pin 5
 *   - Relay 5 (Pump 5): Digital Pin 6
 *
 * Protocol over Serial (115200 baud):
 *   - 5-digit bitmask: "00000" to "11111" (e.g. "00100" turns on Pump 3 only)
 *   - "OFF" or "ALL OFF" or "0" -> Turns off all 5 relays ("00000")
 *   - "ON" or "ALL ON" or "1"   -> Turns on all 5 relays ("11111")
 */

#define BAUDRATE 115200
#define NUM_RELAYS 5

#define RELAY_ON HIGH
#define RELAY_OFF LOW

const int RELAY_PINS[NUM_RELAYS] = {2, 3, 4, 5, 6};

void applyBitmask(const String& mask) {
  for (int i = 0; i < NUM_RELAYS; i++) {
    if (mask[i] == '1') {
      digitalWrite(RELAY_PINS[i], RELAY_ON);
    } else if (mask[i] == '0') {
      digitalWrite(RELAY_PINS[i], RELAY_OFF);
    }
  }
  Serial.print("ACK|RELAYS:");
  Serial.println(mask);
}

void allOff() {
  applyBitmask("00000");
}

void allOn() {
  applyBitmask("11111");
}

void setup() {
  Serial.begin(BAUDRATE);
  for (int i = 0; i < NUM_RELAYS; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    digitalWrite(RELAY_PINS[i], RELAY_OFF);
  }

  Serial.println("READY: 5-Relay Bitmask Controller (Pins 2, 3, 4, 5, 6)");
  Serial.println("Send 5 digits (e.g. '00100') or 'ALL OFF'");
  Serial.println("ACK|RELAYS:00000");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == NUM_RELAYS) {
      // Validate all characters are 0 or 1
      bool valid = true;
      for (int i = 0; i < NUM_RELAYS; i++) {
        if (cmd[i] != '0' && cmd[i] != '1') {
          valid = false;
          break;
        }
      }
      if (valid) {
        applyBitmask(cmd);
      } else {
        Serial.print("ERR|INVALID_MASK:");
        Serial.println(cmd);
      }
    } 
    else if (cmd.equalsIgnoreCase("OFF") || cmd.equalsIgnoreCase("ALL OFF") || cmd.equalsIgnoreCase("ALL_OFF") || cmd == "0") {
      allOff();
    } 
    else if (cmd.equalsIgnoreCase("ON") || cmd.equalsIgnoreCase("ALL ON") || cmd.equalsIgnoreCase("ALL_ON") || cmd == "1") {
      allOn();
    } 
    else if (cmd.equalsIgnoreCase("STATUS")) {
      Serial.print("STATUS|RELAYS:");
      for (int i = 0; i < NUM_RELAYS; i++) {
        Serial.print(digitalRead(RELAY_PINS[i]) == RELAY_ON ? "1" : "0");
      }
      Serial.println();
    }
    else {
      Serial.print("ERR|UNKNOWN_CMD:");
      Serial.println(cmd);
    }
  }
}
