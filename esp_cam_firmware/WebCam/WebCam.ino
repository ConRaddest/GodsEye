#define WIFI_SSID "Galaxy A53 5G CEEF"
#define WIFI_PASS "fbyr2687"
#define HOSTNAME  "esp32cam"

#include <WebServer.h>
#include <eloquent_esp32cam.h>
#include <eloquent_esp32cam/viz/mjpeg.h>

using namespace eloq;
using namespace eloq::viz;

WebServer server(80);

// Function to handle requests for turning on specific LEDs
void handleLedControl() {
    if (server.hasArg("led")) {
        String ledValue = server.arg("led");
        if (ledValue == "-1") {
          // Turning off the flash
          digitalWrite(4, LOW);
          Serial.println("LED on GPIO 4 de-activated");
          server.send(200, "text/plain", "FLASH is OFF (GPIO 4)");
        }
        else if (ledValue == "0") {
          // Turning on the flash
          digitalWrite(4, HIGH);
          Serial.println("LED on GPIO 4 activated");
          server.send(200, "text/plain", "FLASH is ON (GPIO 4)");
        }
        else if (ledValue == "1") {
          // Setting active lighting to Dim
          digitalWrite(12, HIGH);
          digitalWrite(13, LOW);
          digitalWrite(15, LOW);
          Serial.println("LED 1 is ON (GPIO 12)");
          server.send(200, "text/plain", "LED 1 is ON (GPIO 12)");
        } 
        else if (ledValue == "2") {
          // Setting active lighting to Normal
          digitalWrite(12, HIGH);
          digitalWrite(13, HIGH);
          digitalWrite(15, LOW);
          Serial.println("LED 1 & 2 is ON (GPIO 12 & 13)");
          server.send(200, "text/plain", "LED 1 & 2 is ON (GPIO 12 & 13)");
        } 
        else if (ledValue == "3") {
          // Setting active lighting to Bright
          digitalWrite(12, HIGH);
          digitalWrite(13, HIGH);
          digitalWrite(15, HIGH);
          Serial.println("LED 1, 2 & 3 is ON (GPIO 12, 13 & 15)");
          server.send(200, "text/plain", "LED 1, 2 & 3 is ON (GPIO 12, 13 & 15)");
        } 
        else {
          // Invalid input
          server.send(400, "text/plain", "Invalid LED number. Use 1, 2, or 3.");
        }
    } else {
      server.send(400, "text/plain", "No LED number provided.");
    }
}

// Function to handle test connection endpoint, makes camera flash in ackknowledgement
void handleTestConnectionVerbose() {
    Serial.println("Connection test successful");
    server.send(200, "text/plain", "OK");

    digitalWrite(4, HIGH);
    delay(100);
    digitalWrite(4, LOW);
    delay(100);
    digitalWrite(4, HIGH);
    delay(100);
    digitalWrite(4, LOW);
}

// Function to handle test connection endpoint
void handleTestConnection() {
    Serial.println("Connection test successful");
    server.send(200, "text/plain", "OK");
}

void setup() {
  delay(3000);
  Serial.begin(115200);
  Serial.println("___MJPEG STREAM SERVER___");

  // Set GPIO 4, 12, 13, and 15 as output for LED control
  pinMode(4, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
  pinMode(15, OUTPUT);

  // Ensure all LEDs are off initially
  digitalWrite(4, LOW);
  digitalWrite(12, LOW);
  digitalWrite(13, LOW);
  digitalWrite(15, LOW);

  // Camera settings
  camera.pinout.aithinker();
  camera.brownout.disable();
  camera.resolution.uxga();
  camera.quality.high();
  camera.sensor.enableAutomaticWhiteBalance();
  camera.sensor.enableAutomaticGainControl();

  // Init camera
  while (!camera.begin().isOk())
      Serial.println(camera.exception.toString());

  // Connect to WiFi
  while (!wifi.connect().isOk())
      Serial.println(wifi.exception.toString());

  // Start MJPEG HTTP server
  while (!mjpeg.begin().isOk())
      Serial.println(mjpeg.exception.toString());

  // Add endpoints for LED control and connection test
  server.on("/control", handleLedControl); // URL to control LEDs
  server.on("/test_verbose", handleTestConnectionVerbose); // URL for testing the connection
  server.on("/test", handleTestConnection); // URL for testing the connection
  server.begin(); // Start the web server

  Serial.println("Camera OK");
  Serial.println("WiFi OK");
  Serial.println("MjpegStream OK");
  Serial.println(mjpeg.address());

}

void loop() {
    // Handle web server requests
    server.handleClient();
}
