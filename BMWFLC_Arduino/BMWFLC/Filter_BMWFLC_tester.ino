
#include "BMWFLC.h"

//--------------------------------------------------------
//LSM6DS3
#include "SparkFunLSM6DS3.h"
#include "Wire.h"
#include "SimpleTimer.h"

//Create two instances of the driver class
LSM6DS3 SensorOne( I2C_MODE, 0x6A );
LSM6DS3 SensorTwo( I2C_MODE, 0x6B );

SimpleTimer timer;
//Interrupt variables

boolean stop_recording = false;
boolean timer_tick = false;
//--------------------------------------------------------
//Constants
#define MAXK 30  // Maximum sampling instant to count
#define T 0.01 // Sampling time period in second
//--------------------------------------------------------
//External variables
float k = 0; //sampling instant
float s1 = 0; //input signal
float s2 = 0; //input signal
float y = 0; //filtered signal
//--------------------------------------------------------
//define timer
//--------------------------------------------------------
void setup() {
  // initialize serial communications
  Serial.begin(250000);
  delay(1000); //relax...
  //Serial.println("Processor came out of reset.\n");
  InitBMWFLC();//initialize BMFLC

  //Call .begin() to configure the IMUs
  if ( SensorOne.begin() != 0 )
  {
    //Serial.println("Problem starting the sensor at 0x6A.");
  }
  else
  {
    //Serial.println("Sensor at 0x6A started.");
  }
  if ( SensorTwo.begin() != 0 )
  {
    //Serial.println("Problem starting the sensor at 0x6B.");
  }
  else
  {
    //Serial.println("Sensor at 0x6B started.");
  }

  timer.setInterval(10, repeatMe);
}

void repeatMe()
{
  timer_tick = true;
}

void loop() {

  unsigned long start = micros();
  
  timer.run();
  if (timer_tick)
  {
    timer_tick = false;
    
    s1 = SensorTwo.readFloatGyroY() + 5.1;
    s2 = SensorOne.readFloatAccelX();

    //filter
    y = BWMFLC(k, s1);
    k += T;
    //Serial.println(k);
    //plot to serial
    //PlotFrequency();
    //PlotSignal(s)
    //PlotSignal(s2)
    //if(MAXK<k)
    //{stop_recording=true;}
    //PlotSignal(s2);
    //Plot2Signal(s1,s2);
    if (MAXK+0.01 > k)
    {
      PlotSignal(s1);
      //PlotFrequency();
      //PrintAll(s1, s2);
      //Serial.println(k);
        
    }
    
  }
  unsigned long end_t = micros();
  unsigned long delta_t = end_t - start;
  
  if (delta_t > 10000)
  {
    Serial.print("Warning! Loop takes too long."); 
    Serial.println(delta_t); 
  }
  //Serial.println(delta_t); 
}
void PlotFrequency()
{
  Serial.println(get_dominant_frequency_1(), 8);
  //Serial.print(" ");
  //Serial.println(get_dominant_frequency_2(), 8);
}

void PlotSignal(float s)
{
  Serial.println(s, 8);
}

void Plot2Signal(float s1, float s2)
{
  Serial.print(s1, 8);
  Serial.print(" ");
  Serial.println(s2, 8);
}



void PrintAll(float s1, float s2)
{
  Serial.print(s1, 8);
  Serial.print(", ");
  Serial.print(s2, 8);
  Serial.print(", ");
  Serial.print(get_dominant_frequency_1(), 8);
  Serial.print(", ");
  Serial.println(get_dominant_frequency_2(), 8);
}
