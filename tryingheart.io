#include <PulseSensorPlayground.h>  // Includes the PulseSensorPlayground Library.

// Variables
const int PulseWire = 0;           // PulseSensor PURPLE WIRE connected to ANALOG PIN 0
const int LED = LED_BUILTIN;        // The on-board Arduino LED, close to PIN 13.
int Threshold = 550;               // Determine which Signal to "count as a beat" and which to ignore.

PulseSensorPlayground pulseSensor;  // Creates an instance of the PulseSensorPlayground object called "pulseSensor"

int readings[5];                   // Array to store 5 heart rate readings
int sum = 0;                        // Variable to store the sum of readings
float averageHeartRate = 0.0;      // Variable to store the average heart rate
int readingIndex = 0;             // Index to keep track of readings in the array

void setup() {
  Serial.begin(9600);           // For Serial Monitor

  // Configure the PulseSensor object
  pulseSensor.analogInput(PulseWire);
  pulseSensor.blinkOnPulse(LED);
  pulseSensor.setThreshold(Threshold);

  // Double-check the "pulseSensor" object was created and "began" seeing a signal.
  if (pulseSensor.begin()) {
    // Serial.println("We created a pulseSensor Object !");
  }
}

void loop() {
  if (pulseSensor.sawStartOfBeat()) {
    int myBPM = pulseSensor.getBeatsPerMinute();

    // Store the current BPM in the readings array
    readings[readingIndex] = myBPM;
    readingIndex = (readingIndex + 1) % 5;  // Circular buffer approach

    // Check if we have enough readings for average calculation
    if (readingIndex == 0) {
      sum = 0;
      for (int i = 0; i < 5; i++) {
        sum += readings[i];
      }
      averageHeartRate = float(sum) / 5.0;

      // Serial.println("♥ A HeartBeat Happened !");
      // Serial.print("Average BPM: ");
      Serial.println(averageHeartRate);
    } 
    // else {
    //   Serial.println("♥ A HeartBeat Happened !");
    //   Serial.print("BPM: ");
    //   Serial.println(myBPM);
    // }
  }

  delay(2000);  // Considered best practice in a simple sketch.
}
