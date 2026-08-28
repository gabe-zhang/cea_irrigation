void setup() {
  Serial.begin(9600); // open serial port, set the baud rate to 9600 bps
}
void loop() {
  Serial.println(analogRead(A0)); //connect sensor and print the value to serial
  delay(100);
}
