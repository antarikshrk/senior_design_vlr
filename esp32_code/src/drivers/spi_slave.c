/*SPI Slave Implementation File:
    Initializes the SPI Slave
    Handles Byte on MISO Wire
    Sends Command on MOSI Wire
*/
#include "spi_slave.h"
#include "driver/spi_slave.h"

#define MISO 19
#define MOSI 23
#define SCLK 18
#define CS   5

/*Private Variables*/
static volatile esp_status_t current_status = WAIT; //Looks at the current status
static volatile esp_command_t recieved_command = NONE; //If there is no command, set to NONE
static volatile bool new_command_flag = false; //If there is a new command detected
static uint8_t rx_buffer; //RX Buffer 
static uint8_t tx_buffer; //TX Buffer

/*Private Prototypes -> Control Hardware*/
static void spi_hardware_init(void);
static void spi_load_tx_buffer(uint8_t data);

/*Public Functions*/
void spi_slave_init(void) 
{
    recieved_command = NONE;
    new_command_flag = false;
    current_status = WAIT;
    spi_hardware_init();
}

bool spi_recieve_command(void) /*Getter Function*/
{ 
    return new_command_flag; //Set to True
}

esp_command_t spi_get_command(void)
{
    new_command_flag = false; //Once Command has been recieved, set to false
    return recieved_command; //Return the Received Command (For the FSM)
}

esp_status_t spi_slave_set_send_status(esp_status_t status) /*Send the Status to Raspberry Pi*/
{
    current_status = status; //Set the Current Status to the given status
    return current_status; //Return it
}

/*Interrupt Handler - TO DO*/

/*Private Functions*/
static void spi_hardware_init(void) /*FIX*/
{    
    //Configure everything
    spi_bus_config_t bus_cfg = { .mosi_io_num=MOSI, .miso_io_num=MISO, .sclk_io_num=SCLK, .quadwp_io_num=-1, .quadhd_io_num=-1 };
    spi_slave_interface_config_t slave_cfg = { .mode=0, .spics_io_num=CS, .queue_size=1 };
}

static void spi_load_tx_buffer(uint8_t data) /*TO DO*/
{
    //Write data to SPI Transmit Register
}

