/*
 * CEA Irrigation Controller Firmware
 * 
 * Hardware Configuration:
 *   - 0-4 Pump Relays (Digital Output): Pins 2, 3, 4, 5
 *   - DHT22 Temp & Humidity Sensor (Digital Input): Pin 7
 *   - DS18S20 Soil Temperature Sensor (OneWire Digital): Pin 8
 *   - 0-4 Soil Moisture Sensors (Analog Input): Pins A0, A1, A2, A3
 *   - Pan Servo (Digital Output): Pin 9
 *   - Tilt Servo (Digital Output): Pin 10
 *   - Light Sensor (I2C at 0x10): DFRobot VEML7700 Ambient Light Sensor (A4=SDA, A5=SCL)
 * 
 * Servo Range Limits:
 *   - Pan: 0 to 130 deg (Center: 65 deg)
 *   - Tilt: 0 to 90 deg (Center: 60 deg)
 * 
 * Communication Protocol (9600 baud):
 *   - Periodic Telemetry (every 1000ms): Tagged CSV row:
 *     soil,s1,s2,s3,s4,soil_temp,val,temp,val,humi,val,light,val,relays,mask,pan,val,tilt,val
 *     Any missing, disconnected, or unreadable sensor outputs 'null'.
 *   - Control Commands:
 *     - 'p <0-130>' -> Set pan angle (e.g. 'p 65', 'p 0')
 *     - 't <0-90>'  -> Set tilt angle (e.g. 't 60', 't 45')
 *     - 'c'         -> Re-center servos (Pan 65, Tilt 60)
 *     - Bitmask     -> '0' and '1' string up to 4 digits (e.g. '0000', '1000', '01')
 */

#include <Servo.h>
#include <OneWire.h>
#include <dht.h>
#include <Wire.h>
#include <DFRobot_VEML7700.h>

#define BAUDRATE 9600

// Hardware Configuration Counts
#define NUM_RELAYS 4
#define NUM_SOIL   4

#define RELAY_ON  HIGH
#define RELAY_OFF LOW

// Hardware Pin Definitions
const int RELAY_PINS[NUM_RELAYS] = {2, 3, 4, 5};
const int SOIL_PINS[NUM_SOIL]    = {A0, A1, A2, A3};
#define DHT22_PIN   7
#define DS18S20_PIN 8
#define PAN_PIN     9
#define TILT_PIN    10

// Servo Safety Limits & Centers
const int PAN_MIN     = 0;
const int PAN_MAX     = 130;
const int PAN_CENTER  = 65;
const int TILT_MIN    = 0;
const int TILT_MAX    = 90;
const int TILT_CENTER = 60;

Servo panServo;
Servo tiltServo;
int currentPan  = PAN_CENTER;
int currentTilt = TILT_CENTER;

// Servo Idle Auto-Detach Management (prevents interrupt jitter when stationary)
const unsigned long SERVO_DETACH_DELAY = 500; // ms to allow servo to complete movement
unsigned long panMoveStartTime  = 0;
unsigned long tiltMoveStartTime = 0;
bool panActive  = false;
bool tiltActive = false;

void movePan(int angle) {
  angle = constrain(angle, PAN_MIN, PAN_MAX);
  if (!panServo.attached()) {
    panServo.attach(PAN_PIN);
  }
  panServo.write(angle);
  currentPan = angle;
  panActive = true;
  panMoveStartTime = millis();
}

void moveTilt(int angle) {
  angle = constrain(angle, TILT_MIN, TILT_MAX);
  if (!tiltServo.attached()) {
    tiltServo.attach(TILT_PIN);
  }
  tiltServo.write(angle);
  currentTilt = angle;
  tiltActive = true;
  tiltMoveStartTime = millis();
}

void checkServoDetach() {
  unsigned long now = millis();
  if (panActive && (now - panMoveStartTime >= SERVO_DETACH_DELAY)) {
    panServo.detach();
    panActive = false;
  }
  if (tiltActive && (now - tiltMoveStartTime >= SERVO_DETACH_DELAY)) {
    tiltServo.detach();
    tiltActive = false;
  }
}

// Relay States
char relayStates[NUM_RELAYS + 1] = "0000";

// DHT22 State
dht DHT;
double lastValidTemp = 0.0;
double lastValidHumi = 0.0;
bool hasValidDHT     = false;

// Ambient Light Sensor State (DFRobot VEML7700 I2C at 0x10)
DFRobot_VEML7700 als;
float lastValidLux          = 0.0;
bool hasValidLight          = false;
bool lightSensorInitialized = false;

// OneWire DS18S20 Non-blocking State Machine
OneWire ds(DS18S20_PIN);
enum DS18State {
  DS_IDLE,
  DS_CONVERTING
};
DS18State dsState = DS_IDLE;
unsigned long dsConversionStartTime = 0;
unsigned long lastDsSearchTime = 0;
byte dsAddr[8];
bool hasDsAddr = false;
float lastValidSoilTemp = -1000.0;
bool hasValidSoilTemp   = false;

// Telemetry Timing
const unsigned long TELEMETRY_INTERVAL = 1000;
unsigned long lastTelemetryTime = 0;

/**
 * Test if an analog pin has an active sensor connected or is open/floating.
 * Uses the AVR internal pullup: an open pin charges to ~1000-1023 (>= 900).
 * An attached sensor actively drives the pin, yielding a value < 900 (dry air > 750).
 */
bool isSoilSensorAttached(int pin) {
  pinMode(pin, INPUT_PULLUP);
  delayMicroseconds(50);
  int pullVal = analogRead(pin);
  pinMode(pin, INPUT); // restore to high-impedance input
  return (pullVal < 900);
}

/**
 * Non-blocking DS18S20 / DS18B20 temperature read.
 * Triggers conversion and waits >= 750ms without blocking loop().
 */
void updateSoilTemp() {
  unsigned long now = millis();

  if (dsState == DS_IDLE) {
    if (!hasDsAddr) {
      if (now - lastDsSearchTime < 1000) {
        return; // Avoid hammering bus if no sensor is present
      }
      lastDsSearchTime = now;
      if (!ds.search(dsAddr)) {
        ds.reset_search();
        hasDsAddr = false;
        hasValidSoilTemp = false;
        return;
      }
      if (OneWire::crc8(dsAddr, 7) != dsAddr[7] || (dsAddr[0] != 0x10 && dsAddr[0] != 0x28)) {
        hasDsAddr = false;
        hasValidSoilTemp = false;
        ds.reset_search();
        return;
      }
      hasDsAddr = true;
    }

    // Trigger conversion
    ds.reset();
    ds.select(dsAddr);
    ds.write(0x44, 1); // start conversion with parasite power on
    dsConversionStartTime = now;
    dsState = DS_CONVERTING;
  }
  else if (dsState == DS_CONVERTING) {
    // Wait at least 750ms for DS18 conversion to complete
    if (now - dsConversionStartTime >= 750) {
      byte present = ds.reset();
      if (!present) {
        // Sensor unplugged or disconnected
        hasValidSoilTemp = false;
        hasDsAddr = false;
        dsState = DS_IDLE;
        return;
      }

      ds.select(dsAddr);
      ds.write(0xBE); // Read scratchpad

      byte data[9];
      for (int i = 0; i < 9; i++) {
        data[i] = ds.read();
      }

      if (OneWire::crc8(data, 8) == data[8]) {
        int16_t raw = (data[1] << 8) | data[0];
        if (dsAddr[0] == 0x10) {
          // DS18S20 family
          raw = raw << 3;
          if (data[7] == 0x10) {
            raw = (raw & 0xFFF0) + 12 - data[6];
          }
          lastValidSoilTemp = (float)raw * 0.0625;
        } else {
          // DS18B20 family (12-bit, 0.0625 degC/LSB)
          lastValidSoilTemp = (float)raw / 16.0;
        }
        hasValidSoilTemp = true;
      } else {
        hasValidSoilTemp = false;
        hasDsAddr = false;
        ds.reset_search();
      }

      dsState = DS_IDLE;
    }
  }
}

/**
 * Read DHT22 temperature and humidity.
 */
void updateDHT() {
  int chk = DHT.read22(DHT22_PIN);
  if (chk == DHTLIB_OK) {
    lastValidTemp = DHT.temperature;
    lastValidHumi = DHT.humidity;
    hasValidDHT   = true;
  } else {
    hasValidDHT   = false;
  }
}

/**
 * Read ambient light from VEML7700 I2C sensor.
 * Gracefully handles disconnected or unresponsive sensor without hanging.
 */
void updateLight() {
  Wire.beginTransmission(0x10);
  if (Wire.endTransmission() != 0) {
    hasValidLight = false;
    lightSensorInitialized = false;
    return;
  }

  if (!lightSensorInitialized) {
    als.begin();
    lightSensorInitialized = true;
  }

  float lux = 0.0;
  if (als.getALSLux(lux) == DFRobot_VEML7700::STATUS_OK) {
    lastValidLux = lux;
    hasValidLight = true;
  } else {
    hasValidLight = false;
  }
}

/**
 * Output tagged CSV telemetry row:
 * soil,s1,s2,s3,s4,soil_temp,val,temp,val,humi,val,light,val,relays,mask,pan,val,tilt,val
 */
void broadcastTelemetry() {
  // 1-4. Soil moisture channels with 'soil' tag
  Serial.print("soil,");
  for (int i = 0; i < NUM_SOIL; i++) {
    if (isSoilSensorAttached(SOIL_PINS[i])) {
      int val = analogRead(SOIL_PINS[i]);
      Serial.print(val);
    } else {
      Serial.print("null");
    }
    Serial.print(",");
  }

  // 5. Soil temperature
  Serial.print("soil_temp,");
  if (hasValidSoilTemp) {
    Serial.print(lastValidSoilTemp, 1);
  } else {
    Serial.print("null");
  }
  Serial.print(",");

  // 6. Air Temperature
  Serial.print("temp,");
  if (hasValidDHT) {
    Serial.print(lastValidTemp, 1);
  } else {
    Serial.print("null");
  }
  Serial.print(",");

  // 7. Air Humidity
  Serial.print("humi,");
  if (hasValidDHT) {
    Serial.print(lastValidHumi, 1);
  } else {
    Serial.print("null");
  }
  Serial.print(",");

  // 8. Light Sensor
  Serial.print("light,");
  if (hasValidLight) {
    Serial.print(lastValidLux, 1);
  } else {
    Serial.print("null");
  }
  Serial.print(",");

  // 9. Relays Bitmask
  Serial.print("relays,");
  Serial.print(relayStates);
  Serial.print(",");

  // 10. Pan Angle
  Serial.print("pan,");
  Serial.print(currentPan);
  Serial.print(",");

  // 11. Tilt Angle
  Serial.print("tilt,");
  Serial.println(currentTilt);
}

/**
 * Apply bitmask string to relay pins.
 */
void applyBitmask(const String& mask) {
  for (unsigned int i = 0; i < mask.length() && i < NUM_RELAYS; i++) {
    if (mask[i] == '1') {
      digitalWrite(RELAY_PINS[i], RELAY_ON);
      relayStates[i] = '1';
    } else if (mask[i] == '0') {
      digitalWrite(RELAY_PINS[i], RELAY_OFF);
      relayStates[i] = '0';
    }
  }
  Serial.print("ACK: Relays set to ");
  Serial.println(relayStates);
}

/**
 * Handle incoming serial commands.
 */
void handleSerialCommands() {
  if (Serial.available() <= 0) {
    return;
  }

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }

  char firstChar = cmd.charAt(0);

  // Command 'c' or 'C' -> Re-center servos
  if (cmd.equalsIgnoreCase("c")) {
    movePan(PAN_CENTER);
    moveTilt(TILT_CENTER);
    Serial.println("ACK: Re-centered to Pan 65 deg, Tilt 60 deg");
    return;
  }

  // Command 'p <angle>' or 'P <angle>' -> Set Pan
  if (firstChar == 'p' || firstChar == 'P') {
    String angleStr = cmd.substring(1);
    angleStr.trim();
    if (angleStr.length() > 0) {
      int angle = angleStr.toInt();
      movePan(angle);
      Serial.print("ACK: Pan rotated to ");
      Serial.print(currentPan);
      Serial.println(" deg");
    } else {
      Serial.println("ERR: Missing angle for pan. Use 'p <0-130>'");
    }
    return;
  }

  // Command 't <angle>' or 'T <angle>' -> Set Tilt
  if (firstChar == 't' || firstChar == 'T') {
    String angleStr = cmd.substring(1);
    angleStr.trim();
    if (angleStr.length() > 0) {
      int angle = angleStr.toInt();
      moveTilt(angle);
      Serial.print("ACK: Tilt moved to ");
      Serial.print(currentTilt);
      Serial.println(" deg");
    } else {
      Serial.println("ERR: Missing angle for tilt. Use 't <0-90>'");
    }
    return;
  }

  // Bitmask command: 1 to NUM_RELAYS digits of '0' or '1'
  if (cmd.length() >= 1 && cmd.length() <= NUM_RELAYS) {
    bool isBitmask = true;
    for (unsigned int i = 0; i < cmd.length(); i++) {
      if (cmd[i] != '0' && cmd[i] != '1') {
        isBitmask = false;
        break;
      }
    }
    if (isBitmask) {
      applyBitmask(cmd);
      return;
    }
  }

  // Unknown command
  Serial.print("ERR: Unknown command '");
  Serial.print(cmd);
  Serial.println("'. Use 'p <0-130>', 't <0-90>', 'c', or bitmask (e.g. '0000')");
}

void setup() {
  Serial.begin(BAUDRATE);

  // Initialize relay pins
  for (int i = 0; i < NUM_RELAYS; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    digitalWrite(RELAY_PINS[i], RELAY_OFF);
    relayStates[i] = '0';
  }
  relayStates[NUM_RELAYS] = '\0';

  // Initialize, center, and allow pan/tilt servos to settle
  movePan(PAN_CENTER);
  moveTilt(TILT_CENTER);

  // Output format specification on boot
  Serial.println("FORMAT: soil,s1,s2,s3,s4,soil_temp,val,temp,val,humi,val,light,val,relays,mask,pan,val,tilt,val");

  // Initial read delay for sensor stabilization
  delay(500);

  // Initialize I2C bus for sensors
  Wire.begin();

  // Initial sensor sampling
  updateDHT();
  updateSoilTemp();
  updateLight();

  // Initial telemetry broadcast on boot
  broadcastTelemetry();
  lastTelemetryTime = millis();
}

void loop() {
  // Continuously update non-blocking DS18S20 state machine
  updateSoilTemp();

  // Check and detach servos when idle to eliminate interrupt jitter
  checkServoDetach();

  // Periodic telemetry broadcast and sensor sampling
  unsigned long currentMillis = millis();
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = currentMillis;
    updateDHT();
    updateLight();
    broadcastTelemetry();
  }

  // Handle incoming Serial control commands
  handleSerialCommands();
}
