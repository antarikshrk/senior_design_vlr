/*SPI Driver: 
    SPI Slave Initialization
    Handling Recieved Bytes from Raspberry Pi on MISO wire
    Sending Status to the Raspberry Pi on MOSI wire
*/
#ifndef SPI_SLAVE_H
#define SPI_SLAVE_H 

#include <stdint.h>
#include <stdbool.h>

/*Public Types*/
typedef enum
{
    WAIT = 0,
    GO = 1,
    BUSY = 2,
    ERROR = 3
} esp_status_t; //ESP Status

typedef enum
{
    NONE = 0,
    MOVE_L = 1,
    MOVE_R = 2,
    MOVE_D = 3
} esp_command_t; //ESP Command

/*Public API*/
void spi_slave_init(void); //Initialize SPI Slave
esp_status_t spi_slave_set_send_status(esp_status_t status); //Set and Send the Status to Pi
bool spi_receive_command(esp_command_t command); //Confirm command has been receieved
esp_command_t spi_get_command(void); //Get the Command from the Pi


#endif 