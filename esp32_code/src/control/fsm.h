/*FSM Code: Contains IDLE, MOVING_POSITION, DROP_DOOR, ERROR*/
#ifndef FSM_H
#define FSM_H

#include <stdint.h>
#include "spi_slave.h"

/*States*/
typedef enum
{
    IDLE = 0, //Go here when waiting
    MOVING_POSITION, //Move the panels inside
    DROP_DOOR, //Drop, wait for 10 sec, close
    ERROR //Go here only when Interrupt detected
} fsm_state_t;

/*Public API*/
void fsm_init(void);
void fsm_process(uint8_t command);
void fsm_update(uint32_t current_time_ms);


#endif

