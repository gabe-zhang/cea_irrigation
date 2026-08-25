/*
 * CEA Irrigation Controller Firmware
 * 
 * Hardware Configuration:
 *   - 1-5 Pump Relays (Digital Output): Pins 2, 3, 4, 5, 6
 *   - DHT22 Temp & Humidity Sensor (Digital Input): Pin 7
 *   - 1-5 Soil Moisture Sensors (Analog Input): Pins A0, A1, A2, A3, A4
 *   - Ambient Light Sensor (Analog Input): Pin A5
 * 
 * Communication Protocol (115200 baud):
 *   - Telemetry (every 2s): JSON object:
 *     {"soil":[s1,s2,...],"light":val,"temp":24.5,"humidity":60.2,"relays":"0000"}
 *   - Control commands:
 *     - Bitmask only: e.g. "0010", "1111", "0000" (matches NUM_CHANNELS length)
 */

#include <dht.h>

#define BAUDRATE 115200

// Configurable active channel count (1 to 5)
#define NUM_CHANNELS 4

// Enable or disable light sensor on Pin A5 (true: read Pin A5, false: output null)
#define ENABLE_LIGHT_SENSOR false

#define RELAY_ON HIGH
#define RELAY_OFF LOW

// Hardware Pin Definitions
const int RELAY_PINS[5] = {2, 3, 4, 5, 6};
const int SOIL_PINS[5]  = {A0, A1, A2, A3, A4};
#define LIGHT_PIN       A5
#define DHT22_PIN       7

const unsigned long TELEMETRY_INTERVAL = 1000;
unsigned long lastTelemetryTime = 0;

dht DHT;
double lastValidTemp = 0.0;
double lastValidHumi = 0.0;
bool hasValidDHT = false;

// Store current relay states ('0' or '1')
char relayStates[NUM_CHANNELS + 1];

void broadcastTelemetry() {
  // Read soil moisture sensors for active channels
  int soilValues[NUM_CHANNELS];
  for (int i = 0; i < NUM_CHANNELS; i++) {
    soilValues[i] = analogRead(SOIL_PINS[i]);
  }

  // Read DHT22 temperature and humidity
  int chk = DHT.read22(DHT22_PIN);
  if (chk == DHTLIB_OK) {
    lastValidTemp = DHT.temperature;
    lastValidHumi = DHT.humidity;
    hasValidDHT = true;
  }

  // Construct and send JSON string
  Serial.print("{\"soil\":[");
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Serial.print(soilValues[i]);
    if (i < NUM_CHANNELS - 1) {
      Serial.print(",");
    }
  }
  Serial.print("],\"light\":");
  if (ENABLE_LIGHT_SENSOR) {
    int lightVal = analogRead(LIGHT_PIN);
    Serial.print(lightVal);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"temp\":");
  if (hasValidDHT) {
    Serial.print(lastValidTemp, 1);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"humidity\":");
  if (hasValidDHT) {
    Serial.print(lastValidHumi, 1);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"relays\":\"");
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Serial.print(relayStates[i]);
  }
  Serial.println("\"}");
}

void applyBitmask(const String& mask) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    if (mask[i] == '1') {
      digitalWrite(RELAY_PINS[i], RELAY_ON);
      relayStates[i] = '1';
    } else if (mask[i] == '0') {
      digitalWrite(RELAY_PINS[i], RELAY_OFF);
      relayStates[i] = '0';
    }
  }
  // Immediately output updated state
  broadcastTelemetry();
}

void setup() {
  Serial.begin(BAUDRATE);

  // Initialize relay pins
  for (int i = 0; i < NUM_CHANNELS; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    digitalWrite(RELAY_PINS[i], RELAY_OFF);
    relayStates[i] = '0';
  }
  relayStates[NUM_CHANNELS] = '\0';

  // Initial read delay for sensor stabilization
  delay(500);

  // Initial telemetry broadcast on boot
  broadcastTelemetry();
  lastTelemetryTime = millis();
}

void loop() {
  unsigned long currentMillis = millis();

  // Non-blocking periodic telemetry broadcast
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = currentMillis;
    broadcastTelemetry();
  }

  // Handle incoming Serial control commands (bitmask only)
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) {
      return;
    }

    // Direct bitmask command (must match active channel count)
    if (cmd.length() == NUM_CHANNELS) {
      bool valid = true;
      for (int i = 0; i < NUM_CHANNELS; i++) {
        if (cmd[i] != '0' && cmd[i] != '1') {
          valid = false;
          break;
        }
      }
      if (valid) {
        applyBitmask(cmd);
      } else {
        Serial.print("{\"error\":\"INVALID_MASK\",\"received\":\"");
        Serial.print(cmd);
        Serial.println("\"}");
      }
    } else {
      Serial.print("{\"error\":\"INVALID_LENGTH\",\"expected\":");
      Serial.print(NUM_CHANNELS);
      Serial.print(",\"received\":\"");
      Serial.print(cmd);
      Serial.println("\"}");
    }

    // Clear any extra characters left in buffer to prevent stale commands
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}
