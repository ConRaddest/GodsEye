#define WIFI_SSID "The Falling Fig_Ext1"
#define WIFI_PASS "Th3F@ll1ngF1g"
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

// Function to display a two-digit number using LED1 (tens) and LED2 (units)
void displayIPAddress() {
  String mjpegAddress = mjpeg.address();  // Get the full MJPEG address (e.g., "http://192.168.0.107:81")
  
  // Find the start and end of the IP address section
  int startIdx = mjpegAddress.indexOf("://") + 3; // Move past "http://"
  int endIdx = mjpegAddress.indexOf(':', startIdx); // Locate the ':' before the port number
  
  // Extract the IP address part from the MJPEG address
  String ipAddress = mjpegAddress.substring(startIdx, endIdx); // e.g., "192.168.0.107"
  
  // Extract the substring after the last dot, representing the last three digits
  int lastThreeDigits = ipAddress.substring(ipAddress.lastIndexOf('.') + 1).toInt(); // e.g., "107"

  int hundreds = lastThreeDigits / 100;            // Extract the hundreds digit
  int tens = (lastThreeDigits / 10) % 10;          // Extract the tens digit
  int units = lastThreeDigits % 10;                // Extract the units digit

  // Flash LED1 for the tens digit
  for (int i = 0; i < hundreds; i++) {
    digitalWrite(12, HIGH); // Turn on LED1
    delay(300);
    digitalWrite(12, LOW);  // Turn off LED1
    delay(300);
  }

  delay(500); // Pause between tens and units

  // Flash LED2 for the units digit
  for (int i = 0; i < tens; i++) {
    digitalWrite(13, HIGH); // Turn on LED2
    delay(300);
    digitalWrite(13, LOW);  // Turn off LED2
    delay(300);
  }

  delay(500); // Pause between tens and units

  // Flash LED2 for the units digit
  for (int i = 0; i < units; i++) {
    digitalWrite(15, HIGH); // Turn on LED2
    delay(300);
    digitalWrite(15, LOW);  // Turn off LED2
    delay(300);
  }
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

  displayIPAddress();
  delay(1000);
  displayIPAddress();
  delay(1000);
  displayIPAddress();
  delay(1000);

}

void loop() {
    // Handle web server requests
    server.handleClient();
}
