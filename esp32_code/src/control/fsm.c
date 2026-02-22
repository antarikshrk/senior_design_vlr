/*FSM: IDLE, MOVE_POSITION, DROP_DOOR*/
#include "fsm.h"
#include "servo.h"

/*Private Variables*/
static fsm_state_t current_state = IDLE;
static uint32_t state_entry_time = 0;
static uint8_t active_command = 0;


/*Private Functions*/
static void fsm_set_state(fsm_state_t new_state, uint32_t time_ms)
{
    current_state = new_state;
    state_entry_time = time_ms;
}


void fsm_init(void)
{
    current_state = IDLE;
    active_command = 0;
    servo_set_default_positions();
}

void fsm_process_command(uint8_t command)
{
    if (current_state == IDLE)
    {
        if (command >= 1 && command <= 3)
        {
            active_command = command;
            fsm_set_state(MOVING_POSITION, 0);
        }
    }
}

void fsm_update(uint32_t current_time_ms)
{
    switch (current_state)
    {
        case IDLE:
            /* Ensure servos are in idle position */
            servo_set_default_positions();
            break;

        case MOVING_POSITION:

            /* Tell servo driver target based on command */
            servo_move_to_bin(active_command);

            /* Wait until servo finished moving */
            if (!servo_is_busy())
            {
                fsm_set_state(DROP_DOOR, current_time_ms);
            }

            break;

        case DROP_DOOR:

            /* Open trap door */
            servo_open_trapdoor();

            /* Hold for 10 seconds (non-blocking) */
            if ((current_time_ms - state_entry_time) >= 10000)
            {
                servo_close_trapdoor();
                fsm_set_state(IDLE, current_time_ms);
            }

            break;
    }

    /* Update servo driver each loop (for smooth movement) */
    servo_update();
}
