#ifndef HEADER_DEF
#define HEADER_DEF

typedef unsigned char bool;
#define true    1
#define false   0


//MotorDriver
#define IN1A PORTBbits.RB0
#define ENA PORTBbits.RB1
#define IN2A PORTBbits.RB2
#define S1A PORTDbits.RD0

#define IN1B PORTBbits.RB3
#define ENB PORTBbits.RB4
#define IN2B PORTBbits.RB5
#define S1B PORTDbits.RD1


//LEDs
#define LED LATC6

//Ultrasonic Sensor
#define Trigger_Pin PORTDbits.RD2
#define Echo_Pin PORTDbits.RD3

#endif