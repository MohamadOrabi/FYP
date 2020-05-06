#include "Configurations_Header_File.h"  /*Header file for Configuration Bits*/
#include <stdio.h>
#include <pic18f4550.h>                  /*Header file PIC18f4550 definitions*/

#include "def.h"
#include "functions.h"

#define _XTAL_FREQ 8000000      /* define macros to use internal __delay_us function */
#include <xc.h>

int xa = 0;
int xb = 0;

//int CHORDS_LENGTH = 211;
const int Distance_to_keep = 20;
const  int Error_allowed = 5;

char out;

void main()
{   
    Initialize();
    
    float Distance;
   
    while(1){
        //char x = 0xF0;
        //USART_TxChar(x);
        //MSdelay(500);
    }
    while (1){
        
        float Distance = getDistance();
        
        if (Distance > Distance_to_keep + Error_allowed ){
            //If target is too far
            LED = 1;
            KeepForward();
            //Backwards(1);   
        } else if ( Distance < Distance_to_keep - Error_allowed ) {
            //If target is too close
            LED = 1;
            KeepBackwards();
        } else {
            Stop();
            LED = 0;
        }
    }
}

void __interrupt () ISR (void) {

    if(RCIF==1){
        if (CREN == 0) {    // Check if error occurred
            CREN = 1;   //Clear error
        }
        out=RCREG;	/* Copy received data from RCREG register */
    }
}