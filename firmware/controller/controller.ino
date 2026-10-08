/*
 * CEA Irrigation Controller Firmware
 * Multi-sensor telemetry + relay/servo control over serial (9600 baud, 1s interval).
 */

#include <Servo.h>
#include <OneWire.h>
#include <dht.h>
#include <Wire.h>
#include <DFRobot_VEML7700.h>
#include "PumpSafety.h"
#include "SafetyStorage.h"
#include <EEPROM.h>

#define BAUDRATE 9600

#define NUM_RELAYS 4
#define NUM_SOIL   4

#define RELAY_ON  HIGH
#define RELAY_OFF LOW

const int RELAY_PINS[NUM_RELAYS] = {2, 3, 4, 5};
const int SOIL_PINS[NUM_SOIL]    = {A0, A1, A2, A3};
#define DHT22_PIN   7
#define DS18S20_PIN 8

// ── Servo Configuration ──────────────────────────────────────────────
const unsigned long SERVO_DETACH_DELAY = 1500; // ms before auto-detach (prevents interrupt jitter)

struct ServoConfig {
  Servo         servo;
  int           pin, minAngle, maxAngle, homeAngle, current;
  unsigned long moveTime;
  bool          active;
};

#define PAN  0
#define TILT 1
ServoConfig servos[] = {
  { Servo(),  9, 0, 130, 55, 55, 0, false },  // Pan:  0-130°, home 55°
  { Servo(), 10, 0,  60, 30, 30, 0, false },  // Tilt: 0-60°,  home 30°
};

void moveServo(int idx, int angle) {
  ServoConfig& s = servos[idx];
  angle = constrain(angle, s.minAngle, s.maxAngle);
  // Write before attach: pre-loads the target pulse into the Servo library's
  // internal state so the servo goes directly to 'angle' on re-attach instead
  // of snapping to a default/stale position first (critical for gravity-loaded axes).
  if (!s.servo.attached()) {
    s.servo.write(angle);   // pre-load target
    s.servo.attach(s.pin);  // enable PWM at that angle
  } else {
    s.servo.write(angle);
  }
  s.current  = angle;
  s.active   = true;
  s.moveTime = millis();
}

void checkServoDetach() {
  unsigned long now = millis();
  for (int i = 0; i < 2; i++) {
    if (servos[i].active && (now - servos[i].moveTime >= SERVO_DETACH_DELAY)) {
      servos[i].servo.detach();
      servos[i].active = false;
    }
  }
}

// ── Relay State ──────────────────────────────────────────────────────
char relayStates[NUM_RELAYS + 1] = "0000";
PumpGroupSafety pumpGroup;
SafetyStorage<EEPROMClass> safetyStorage(EEPROM);

void updateRelay(int idx) {
  digitalWrite(RELAY_PINS[idx], pumpGroup.pumps[idx].on ? RELAY_ON : RELAY_OFF);
  relayStates[idx] = pumpGroup.pumps[idx].on ? '1' : '0';
}

bool persistSafety() {
  if (safetyStorage.save(pumpGroup.record)) return true;
  pumpGroup.trip(3, 0);
  for (int i = 0; i < NUM_RELAYS; ++i) updateRelay(i);
  Serial.println(F("ERR: Safety storage failed; all pumps locked"));
  return false;
}

void checkPumpSafety() {
  uint32_t before = pumpGroup.record.generation;
  pumpGroup.check(millis());
  for (int i = 0; i < NUM_RELAYS; i++) updateRelay(i);
  // Physical outputs are OFF before any potentially slow EEPROM writes.
  if (pumpGroup.record.generation != before) persistSafety();
}

// ── DHT22 ────────────────────────────────────────────────────────────
dht DHT;
double lastValidTemp = 0.0;
double lastValidHumi = 0.0;
bool hasValidDHT     = false;

// ── VEML7700 Ambient Light (I2C 0x10) ───────────────────────────────
DFRobot_VEML7700 als;
float lastValidLux          = 0.0;
bool hasValidLight          = false;
bool lightSensorInitialized = false;

// ── DS18S20 Non-blocking State Machine ──────────────────────────────
OneWire ds(DS18S20_PIN);
enum DS18State { DS_IDLE, DS_CONVERTING };
DS18State dsState = DS_IDLE;
unsigned long dsConversionStartTime = 0;
unsigned long lastDsSearchTime = 0;
byte dsAddr[8];
bool hasDsAddr = false;
float lastValidSoilTemp = -1000.0;
bool hasValidSoilTemp   = false;

// ── Telemetry Timing ─────────────────────────────────────────────────
const unsigned long TELEMETRY_INTERVAL = 1000;
unsigned long lastTelemetryTime = 0;

/**
 * Test if an analog pin has an active sensor via internal pullup.
 * Open/floating pins read >= 900; driven pins read < 900.
 */
bool isSoilSensorAttached(int pin) {
  pinMode(pin, INPUT_PULLUP);
  delayMicroseconds(50);
  int pullVal = analogRead(pin);
  pinMode(pin, INPUT);
  return (pullVal < 900);
}

/**
 * Non-blocking DS18S20/DS18B20 temperature read.
 * Triggers conversion and reads after >= 750ms without blocking loop().
 */
void updateSoilTemp() {
  unsigned long now = millis();

  if (dsState == DS_IDLE) {
    if (!hasDsAddr) {
      if (now - lastDsSearchTime < 1000) return; // throttle bus searches
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

    ds.reset();
    ds.select(dsAddr);
    ds.write(0x44, 1); // start conversion with parasite power
    dsConversionStartTime = now;
    dsState = DS_CONVERTING;
  }
  else if (dsState == DS_CONVERTING) {
    if (now - dsConversionStartTime >= 750) {
      byte present = ds.reset();
      if (!present) {
        // Sensor disconnected
        hasValidSoilTemp = false;
        hasDsAddr = false;
        dsState = DS_IDLE;
        return;
      }

      ds.select(dsAddr);
      ds.write(0xBE); // read scratchpad

      byte data[9];
      for (int i = 0; i < 9; i++) data[i] = ds.read();

      if (OneWire::crc8(data, 8) == data[8]) {
        int16_t raw = (data[1] << 8) | data[0];
        if (dsAddr[0] == 0x10) {
          // DS18S20: extended resolution
          raw = raw << 3;
          if (data[7] == 0x10) raw = (raw & 0xFFF0) + 12 - data[6];
          lastValidSoilTemp = (float)raw * 0.0625;
        } else {
          // DS18B20: 12-bit, 0.0625 °C/LSB
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
 * Read VEML7700 lux. Gracefully handles disconnected sensor.
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

// ── Telemetry ────────────────────────────────────────────────────────

/** Print "tag,value," or "tag,null," for a float sensor reading. */
void printTaggedValue(const char* tag, bool valid, float value) {
  Serial.print(tag);
  Serial.print(',');
  if (valid) Serial.print(value, 1);
  else       Serial.print("null");
  Serial.print(',');
}

/**
 * Output tagged CSV: soil,s1..s4,soil_temp,v,temp,v,humi,v,light,v,relays,mask,pan,v,tilt,v
 */
void broadcastTelemetry() {
  Serial.print("soil,");
  for (int i = 0; i < NUM_SOIL; i++) {
    if (isSoilSensorAttached(SOIL_PINS[i])) Serial.print(analogRead(SOIL_PINS[i]));
    else                                     Serial.print("null");
    Serial.print(",");
  }

  printTaggedValue("soil_temp", hasValidSoilTemp, lastValidSoilTemp);
  printTaggedValue("temp",      hasValidDHT,      lastValidTemp);
  printTaggedValue("humi",      hasValidDHT,      lastValidHumi);
  printTaggedValue("light",     hasValidLight,     lastValidLux);

  Serial.print("relays,");
  Serial.print(relayStates);
  Serial.print(",pan,");
  Serial.print(servos[PAN].current);
  Serial.print(",tilt,");
  Serial.print(servos[TILT].current);
  Serial.print(F(",safety_version,1,safety_locked,"));
  Serial.print(pumpGroup.record.locked ? 1 : 0);
  Serial.print(F(",safety_generation,"));
  Serial.print(pumpGroup.record.generation);
  Serial.print(F(",safety_source,"));
  Serial.print(pumpGroup.record.source);
  Serial.print(F(",safety_channels,"));
  Serial.println(pumpGroup.record.channels);
}

// ── Commands ─────────────────────────────────────────────────────────

void applyBitmask(const String& mask) {
  checkPumpSafety();
  uint32_t before = pumpGroup.record.generation;
  if (pumpGroup.record.locked && mask.indexOf('1') >= 0) {
    Serial.println(F("ERR: All pumps locked; inspect the issue and use safety resolve <generation>"));
    return;
  }
  for (unsigned int i = 0; i < mask.length() && i < NUM_RELAYS; i++) {
    pumpGroup.request(i, mask[i] == '1', millis());
  }
  for (int i = 0; i < NUM_RELAYS; ++i) updateRelay(i);
  if (before != pumpGroup.record.generation) persistSafety();
  Serial.print("ACK: Relays set to ");
  Serial.println(relayStates);
}

void handleSerialCommands() {
  // A missing newline or continuous serial traffic must not block safety checks.
  static char buffer[32];
  static byte length = 0;
  static bool overflow = false;
  bool complete = false;
  for (byte consumed = 0; consumed < sizeof(buffer) && Serial.available() > 0; consumed++) {
    char c = Serial.read();
    if (c == '\n') { complete = true; break; }
    if (length < sizeof(buffer) - 1) buffer[length++] = c;
    else overflow = true;
  }
  if (!complete) return;
  buffer[length] = '\0';
  String cmd = overflow ? String("") : String(buffer);
  length = 0;
  overflow = false;
  cmd.trim();
  if (cmd.length() == 0) return;

  char firstChar = cmd.charAt(0);

  if (cmd == "safety status") { broadcastTelemetry(); return; }
  if (cmd.startsWith("safety trip ")) {
    String value = cmd.substring(12);
    if (value.length() == 0 || value.length() > 2) { Serial.println(F("ERR: Invalid safety channel mask")); return; }
    for (unsigned int i = 0; i < value.length(); ++i)
      if (value[i] < '0' || value[i] > '9') { Serial.println(F("ERR: Invalid safety channel mask")); return; }
    int channels = value.toInt();
    if (channels > 15) { Serial.println(F("ERR: Invalid safety channel mask")); return; }
    uint32_t before = pumpGroup.record.generation;
    pumpGroup.trip(2, uint8_t(channels));
    for (int i = 0; i < NUM_RELAYS; ++i) updateRelay(i);
    if (before != pumpGroup.record.generation) persistSafety();
    broadcastTelemetry();
    return;
  }
  if (cmd.startsWith("safety resolve ")) {
    String value = cmd.substring(15);
    if (value.length() == 0 || value.length() > 10) { Serial.println(F("ERR: Invalid safety generation")); return; }
    for (unsigned int i = 0; i < value.length(); ++i)
      if (value[i] < '0' || value[i] > '9') { Serial.println(F("ERR: Invalid safety generation")); return; }
    // Compare decimal text too, so overflow cannot turn a stale value into a match.
    uint32_t generation = strtoul(value.c_str(), NULL, 10);
    if (String(generation) != value || !pumpGroup.resolve(generation)) {
      Serial.println(F("ERR: Safety reset rejected; refresh safety status"));
    } else if (persistSafety()) {
      Serial.println(F("ACK: Safety resolved; all pumps remain OFF"));
    }
    broadcastTelemetry();
    return;
  }

  // Home servos (supports 'h' or legacy 'c')
  if (cmd.equalsIgnoreCase("h") || cmd.equalsIgnoreCase("c")) {
    moveServo(PAN,  servos[PAN].homeAngle);
    moveServo(TILT, servos[TILT].homeAngle);
    Serial.println("ACK: Homed to Pan 55 deg, Tilt 30 deg");
    return;
  }

  // Pan or Tilt angle command
  int servoIdx = -1;
  if (firstChar == 'p' || firstChar == 'P') servoIdx = PAN;
  if (firstChar == 't' || firstChar == 'T') servoIdx = TILT;
  if (servoIdx >= 0) {
    String angleStr = cmd.substring(1);
    angleStr.trim();
    if (angleStr.length() > 0) {
      moveServo(servoIdx, angleStr.toInt());
      const char* name = (servoIdx == PAN) ? "Pan rotated" : "Tilt moved";
      Serial.print("ACK: ");
      Serial.print(name);
      Serial.print(" to ");
      Serial.print(servos[servoIdx].current);
      Serial.println(" deg");
    } else {
      if (servoIdx == PAN) Serial.println("ERR: Missing angle for pan. Use 'p <0-130>'");
      else                 Serial.println("ERR: Missing angle for tilt. Use 't <0-60>'");
    }
    return;
  }

  // Relay bitmask: 1-4 digits of '0'/'1'
  if (cmd.length() >= 1 && cmd.length() <= NUM_RELAYS) {
    bool isBitmask = true;
    for (unsigned int i = 0; i < cmd.length(); i++) {
      if (cmd[i] != '0' && cmd[i] != '1') { isBitmask = false; break; }
    }
    if (isBitmask) { applyBitmask(cmd); return; }
  }

  Serial.print("ERR: Unknown command '");
  Serial.print(cmd);
  Serial.println("'. Use 'p <0-130>', 't <0-60>', 'h', 'c', or bitmask (e.g. '0000')");
}

// ── Setup & Loop ─────────────────────────────────────────────────────

void setup() {
  Serial.begin(BAUDRATE);

  for (int i = 0; i < NUM_RELAYS; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    digitalWrite(RELAY_PINS[i], RELAY_OFF);
    relayStates[i] = '0';
  }
  relayStates[NUM_RELAYS] = '\0';

  pumpGroup.record = safetyStorage.load();

  moveServo(PAN,  servos[PAN].homeAngle);
  moveServo(TILT, servos[TILT].homeAngle);

  Serial.println(F("FORMAT: soil,s1,s2,s3,s4,soil_temp,val,temp,val,humi,val,light,val,relays,mask,pan,val,tilt,val,safety_version,val,safety_locked,val,safety_generation,val,safety_source,val,safety_channels,val"));

  delay(500);
  Wire.begin();
  Wire.setWireTimeout(25000, true); // Bound I2C stalls so relay safety keeps running.

  updateDHT();
  updateSoilTemp();
  updateLight();
  broadcastTelemetry();
  lastTelemetryTime = millis();
}

void loop() {
  checkPumpSafety();
  updateSoilTemp();
  checkServoDetach();

  unsigned long currentMillis = millis();
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = currentMillis;
    updateDHT();
    updateLight();
    broadcastTelemetry();
  }

  handleSerialCommands();
  checkPumpSafety();
}
