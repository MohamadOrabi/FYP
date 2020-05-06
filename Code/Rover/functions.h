#ifndef HEADER_FUNCTIONS
#define HEADER_FUNCTIONS

//Global Variables
extern int xa;
extern int xb;

//Initialize
void Initialize();

//Delay
void MSdelay(unsigned int);

//Movement
void KeepForward();
void KeepBackwards();
void KeepRight();
void KeepLeft();
void Stop();

void Forward(unsigned char);
void Backwards(unsigned char);
void Right(unsigned char);
void Left(unsigned char);

//Ultrasonic Sensor
#define _XTAL_FREQ 8000000
void Trigger_Pulse_10us(void);
float getDistance(void);

//Serial Communication
void USART_TxChar(char);

#endif