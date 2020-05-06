#include <xc.h>
#include "functions.h"
#include "def.h"
//Delay
void MSdelay(unsigned int val) {
 unsigned int i,j;
 for(i=0;i<val;i++)
     for(j=0;j<165;j++);         /*This count Provide delay of 1 ms for 8MHz Frequency */
 } 

#define baud_rate 9600
#define F_CPU 8000000/64
#define Baud_value (((float)(F_CPU)/(float)baud_rate)-1)
//Initialize
void Initialize() {
    OSCCON=0x72;                /* Use internal oscillator of 8MHz Frequency */
    
    TRISB=0x00;             //PortB Motor Driver Output
    
    //LED
    TRISCbits.TRISC6 = 0;   //LED Output
    
    //Ultrasonic
    TRISDbits.TRISD2 = 0;   //Trigger Output
    TRISDbits.TRISD3 = 1;   //Echo Input
    
    //Timer
    T1CON = 0x80;
    TMR1IF = 0;			/* Make Timer1 Overflow Flag to '0' */
    TMR1=0;			/* Load Timer1 with 0 */
    
    //Serial Communication
    TRISCbits.TRISC7 = 1;
    TRISCbits.TRISC6 = 0;
    
    float temp;
    temp=Baud_value;     
    SPBRG=(int)temp;                /*baud rate=9600, SPBRG = (F_CPU /(64*9600))-1*/

    TXSTA = 0x20;  	/* Enable Transmit(TX) */ 
    RCSTA = 0x90;  	/* Enable Receive(RX) & Serial */
    ABDEN = 0;

    SYNC = 0;
    SPEN = 1;
    CREN = 1;
    
    INTCONbits.GIE = 1;	/* Enable Global Interrupt */
    INTCONbits.PEIE = 1;/* Enable Peripheral Interrupt */
    PIE1bits.RCIE = 1;	/* Enable Receive Interrupt*/
    PIE1bits.TXIE = 0;	/* Enable Transmit Interrupt*/


     
    //LED = 1;
}

//Movement
const int CHORDS_LENGTH = 1;

void KeepForward(){
    ENA = 1;
    IN1A = 1;
    IN2A = 0;

    ENB = 1;
    IN1B = 1;
    IN2B = 0;
}

void KeepBackwards(){
    ENA = 1;
    IN1A = 0;
    IN2A = 1;

    ENB = 1;
    IN1B = 0;
    IN2B = 1;
}

void KeepRight(){
    ENA = 1;
    IN1A = 1;
    IN2A = 0;

    ENB = 1;
    IN1B = 0;
    IN2B = 1;
}

void KeepLeft(){
    ENA = 1;
    IN1A = 0;
    IN2A = 1;

    ENB = 1;
    IN1B = 1;
    IN2B = 0;
}

void Stop(){
    ENA = 1;
    IN1A = 0;
    IN2A = 0;

    ENB = 1;
    IN1B = 0;
    IN2B = 0;
}

void Forward(unsigned char chords){
    bool done = false;
    bool S1A_old = S1A;
    bool S1B_old = S1B;
    
    int initial_xa = xa;
    int initial_xb = xb;
    
    KeepForward();

    while (!done) {
        
        //Edge Detection
        if (S1A != S1A_old){
            xa++;
            S1A_old = S1A;
        }
        
        if (S1B != S1B_old){
            xb++;
            S1B_old = S1B;
        }
        
        //Check Done flag
        if (xa >= initial_xa + chords * CHORDS_LENGTH || xb >= initial_xb + chords * CHORDS_LENGTH){
            done = true;
        }
        

    }
    Stop();
}

void Backwards(unsigned char chords){
 bool done = false;
    bool S1A_old = S1A;
    bool S1B_old = S1B;
    
    int initial_xa = xa;
    int initial_xb = xb;
    
    KeepBackwards();

    while (!done) {
        
        //Edge Detection
        if (S1A != S1A_old){
            xa++;
            S1A_old = S1A;
        }
        
        if (S1B != S1B_old){
            xb++;
            S1B_old = S1B;
        }
        
        //Check Done flag
        if (xa >= initial_xa + chords * CHORDS_LENGTH || xb >= initial_xb + chords * CHORDS_LENGTH){
            done = true;
        }
        

    }
    
    Stop();
}

void Right(unsigned char chords){

    bool done = false;
    bool S1A_old = S1A;
    bool S1B_old = S1B;
    
    int initial_xa = xa;
    int initial_xb = xb;
    
    KeepRight();

    while (!done) {
        
        //Edge Detection
        if (S1A != S1A_old){
            xa++;
            S1A_old = S1A;
        }
        
        if (S1B != S1B_old){
            xb++;
            S1B_old = S1B;
        }
        
        //Check Done flag
        if (xa >= initial_xa + chords * CHORDS_LENGTH || xb >= initial_xb + chords * CHORDS_LENGTH){
            done = true;
        }
        

    }
    
    Stop();
}

void Left(unsigned char chords){

    bool done = false;
    bool S1A_old = S1A;
    bool S1B_old = S1B;
    
    int initial_xa = xa;
    int initial_xb = xb;
    
    KeepLeft();

    while (!done) {
        
        //Edge Detection
        if (S1A != S1A_old){
            xa++;
            S1A_old = S1A;
        }
        
        if (S1B != S1B_old){
            xb++;
            S1B_old = S1B;
        }
        
        //Check Done flag
        if (xa >= initial_xa + chords * CHORDS_LENGTH || xb >= initial_xb + chords * CHORDS_LENGTH){
            done = true;
        }
        

    }
    
    Stop();
}


//Ultrasonic
void Trigger_Pulse_10us() {
    Trigger_Pin = 1;
    __delay_us(10);
    Trigger_Pin = 0;
}

float getDistance(){
    float Distance;
    unsigned int Time;
    
    Trigger_Pulse_10us();
    while(Echo_Pin == 0); 
    TMR1ON=1;			/* Turn ON Timer1*/
    while(Echo_Pin==1 && !TMR1IF);/* Wait for falling edge */
    Time = TMR1;		/* Copy Time when echo is received */
    TMR1ON=0;			/* Turn OFF Timer1 */
    TMR1=0;			/* Load Timer1 register with 0 */
    if (TMR1IF == 1){
        Distance = 999;
        TMR1IF = 0;			/* Make Timer1 Overflow Flag to '0' */
    } else {
        Distance = ((float)Time/117.00);/* Distance =(velocity x Time)/2 */
    }
    return Distance;
}

//Serial Communication
void USART_TxChar(char out) {        
        //while(TXIF==0);            /*wait for transmit interrupt flag*/
        TXREG=out;                 /*transmit data via TXREG register*/    
}