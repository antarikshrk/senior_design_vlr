/*
 * spi_slave.ino  –  ESP32-C6  SPI Slave  +  FSM  +  LEDC PWM
 *
 * SPI pins (ESP32-C6 default FSPI / SPI2):
 * SCLK  -> GPIO 6   (Pi SCLK / BCM 11)
 * MOSI  -> GPIO 7   (Pi MOSI / BCM 10)
 * MISO  -> GPIO 5   (Pi MISO / BCM  9)
 * CS    -> GPIO 1   (Pi CE0  / BCM  8)
 *
 * Servo pins (LEDC PWM, 50 Hz, 10-bit):
 * Servo 1  -> GPIO 3
 * Servo 2  -> GPIO 2
 * Servo 3  -> GPIO 11
 * Servo 4  -> GPIO 10
 *
 * Packet format (32-byte frame, Pi -> ESP32):
 * [0]  Header   0xAA
 * [1]  Command  0x00=NON | 0x01=BIN_1 | 0x02=BIN_2 | 0x03=BIN_3
 * [2]  Checksum XOR of bytes [0] and [1]
 * [3-31] padding 0x00
 *
 * Bin / command mapping:
 * BIN_1 (0x01) – WHITE   / COTTON   (servo left  position)
 * BIN_2 (0x02) – (n/a)  / DENIM    (servo centre position)
 * BIN_3 (0x03) – NON_WHITE / FLEECE (servo right position)
 *
 * Status byte returned on MISO byte [0]:
 * 0x00=READY  0x01=BUSY  0x02=ERROR  0x03=DONE
 *
 * Library: ESP32 Arduino core >= 3.0.0 by Espressif
 */

#include "driver/ledc.h"
#include "driver/spi_slave.h"
#include "driver/gpio.h"

// ── Pin definitions ──────────────────────────────────────────────────────────
#define SPI_SCLK_PIN   6
#define SPI_MOSI_PIN   7
#define SPI_MISO_PIN   5
#define SPI_CS_PIN     1

#define SERVO1_PIN  3
#define SERVO2_PIN  2
#define SERVO3_PIN  11
#define SERVO4_PIN  10

// ── PWM / servo ──────────────────────────────────────────────────────────────
#define PWM_FREQ       50
#define PWM_RESOLUTION LEDC_TIMER_10_BIT
#define PERIOD_US      20000

#define CH_SERVO1  LEDC_CHANNEL_0
#define CH_SERVO2  LEDC_CHANNEL_1
#define CH_SERVO3  LEDC_CHANNEL_2
#define CH_SERVO4  LEDC_CHANNEL_3

// ── Custom Servo Configurations ──────────────────────────────────────────────
struct ServoConfig {
    int max_deg;
    int min_us;
    int max_us;
};

// Maps to channels 0 through 3 respectively
const ServoConfig servoConfigs[4] = {
    {199, 556, 2420}, // SERVO 1 (CH_SERVO1 / LEDC_CHANNEL_0)
    {270, 500, 2500}, // SERVO 2 (CH_SERVO2 / LEDC_CHANNEL_1)
    {199, 556, 2420}, // SERVO 3 (CH_SERVO3 / LEDC_CHANNEL_2)
    {270, 500, 2500}  // SERVO 4 (CH_SERVO4 / LEDC_CHANNEL_3)
};

// ── SPI protocol ─────────────────────────────────────────────────────────────
#define SPI_FRAME_LEN  32
#define PKT_HEADER     0xAA

#define STATUS_READY   0x00
#define STATUS_BUSY    0x01
#define STATUS_ERROR   0x02
#define STATUS_DONE    0x03

#define CMD_NON    0x00
#define CMD_BIN_1  0x01   // WHITE  / COTTON
#define CMD_BIN_2  0x02   // DENIM  (fabric mode only)
#define CMD_BIN_3  0x03   // NON_WHITE / FLEECE

// ── Servo starting positions ─────────────────────────────────────────────────
int servopos1 = 90;
int servopos2 = 90;
int servopos3 = 90;
int servopos4 = 90;

unsigned long moveStartTime = 0;
unsigned long dropStartTime = 0;

// ── FSM ──────────────────────────────────────────────────────────────────────
enum class State  { IDLE, MOVE, DROP, DONE, ERROR };
enum class BinCmd { BIN_1, BIN_2, BIN_3 };

volatile State   currentState  = State::IDLE;
volatile uint8_t currentStatus = STATUS_READY;
BinCmd           detectedBin   = BinCmd::BIN_1;
volatile bool    detected      = false;

static WORD_ALIGNED_ATTR uint8_t spi_rx_buf[SPI_FRAME_LEN];
static WORD_ALIGNED_ATTR uint8_t spi_tx_buf[SPI_FRAME_LEN];
static spi_slave_transaction_t spi_trans;


// ── Servo helpers ─────────────────────────────────────────────────────────────
uint32_t angleToDuty(ledc_channel_t ch, int angle) {
    int idx = (int)ch; // LEDC_CHANNEL_0 is 0, etc.
    const ServoConfig& cfg = servoConfigs[idx];

    angle = constrain(angle, 0, cfg.max_deg);
    uint32_t pulseUs = (uint32_t)map(angle, 0, cfg.max_deg, cfg.min_us, cfg.max_us);
    
    // Convert target microseconds to a 10-bit duty cycle (0-1023)
    return (pulseUs * 1024UL) / PERIOD_US;
}

void servoWrite(ledc_channel_t ch, int angle) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, ch, angleToDuty(ch, angle));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ch);
}

void setupLEDC() {
    ledc_timer_config_t timer = {};
    timer.speed_mode      = LEDC_LOW_SPEED_MODE;
    timer.timer_num       = LEDC_TIMER_0;
    timer.duty_resolution = PWM_RESOLUTION;
    timer.freq_hz         = PWM_FREQ;
    timer.clk_cfg         = LEDC_AUTO_CLK;
    ledc_timer_config(&timer);

    auto attachServo = [](ledc_channel_t ch, int pin) {
        ledc_channel_config_t cfg = {};
        cfg.gpio_num   = pin;
        cfg.speed_mode = LEDC_LOW_SPEED_MODE;
        cfg.channel    = ch;
        cfg.timer_sel  = LEDC_TIMER_0;
        cfg.duty       = 0;
        cfg.hpoint     = 0;
        cfg.intr_type  = LEDC_INTR_DISABLE;
        ledc_channel_config(&cfg);
    };

    attachServo(CH_SERVO1, SERVO1_PIN);
    attachServo(CH_SERVO2, SERVO2_PIN);
    attachServo(CH_SERVO3, SERVO3_PIN);
    attachServo(CH_SERVO4, SERVO4_PIN);

    servoWrite(CH_SERVO1, servopos1);
    servoWrite(CH_SERVO2, servopos2);
    servoWrite(CH_SERVO3, servopos3);
    servoWrite(CH_SERVO4, servopos4);
}

// ── SPI setup ────────────────────────────────────────────────────────────────
void setupSPISlave() {
    spi_bus_config_t bus_cfg = {};
    bus_cfg.mosi_io_num     = SPI_MOSI_PIN;
    bus_cfg.miso_io_num     = SPI_MISO_PIN;
    bus_cfg.sclk_io_num     = SPI_SCLK_PIN;
    bus_cfg.quadwp_io_num   = -1;
    bus_cfg.quadhd_io_num   = -1;
    bus_cfg.max_transfer_sz = SPI_FRAME_LEN;

    spi_slave_interface_config_t slave_cfg = {};
    slave_cfg.mode          = 0;
    slave_cfg.spics_io_num  = SPI_CS_PIN;
    slave_cfg.queue_size    = 1;
    slave_cfg.flags         = 0;
    slave_cfg.post_setup_cb = NULL;
    slave_cfg.post_trans_cb = NULL;

    ESP_ERROR_CHECK(spi_slave_initialize(SPI2_HOST, &bus_cfg, &slave_cfg, SPI_DMA_CH_AUTO));
    Serial.println("[SPI] Slave initialised on SPI2_HOST (FSPI)");
}

// ── Packet parsing ────────────────────────────────────────────────────────────
bool parsePacket(const uint8_t* buf, uint8_t* cmd_out) {
    if (buf[0] != PKT_HEADER) return false;
    uint8_t expected_cksum = buf[0] ^ buf[1];
    if (buf[2] != expected_cksum) return false;
    *cmd_out = buf[1];
    return true;
}

// ── SPI poll (called every loop) ─────────────────────────────────────────────
void pollSPI() {
    memset(spi_tx_buf, 0x00, SPI_FRAME_LEN);
    spi_tx_buf[0] = currentStatus;

    memset(&spi_trans, 0, sizeof(spi_trans));
    spi_trans.length    = SPI_FRAME_LEN * 8;
    spi_trans.rx_buffer = spi_rx_buf;
    spi_trans.tx_buffer = spi_tx_buf;

    esp_err_t queue_err = spi_slave_queue_trans(SPI2_HOST, &spi_trans, 0);
    if (queue_err != ESP_OK) return;

    spi_slave_transaction_t* result = nullptr;
    esp_err_t get_err = spi_slave_get_trans_result(SPI2_HOST, &result, 0);
    if (get_err != ESP_OK || result == nullptr) return;

    uint8_t cmd = CMD_NON;
    if (parsePacket(spi_rx_buf, &cmd)) {
        Serial.printf("[SPI] Valid packet: cmd=0x%02X\n", cmd);
        if (currentState == State::IDLE) {
            if (cmd == CMD_BIN_1) {
                detectedBin = BinCmd::BIN_1;
                detected    = true;
            } else if (cmd == CMD_BIN_2) {
                detectedBin = BinCmd::BIN_2;
                detected    = true;
            } else if (cmd == CMD_BIN_3) {
                detectedBin = BinCmd::BIN_3;
                detected    = true;
            }
            // CMD_NON (0x00) is ignored – status poll only
        }
    } else {
        Serial.printf("[SPI] Bad packet: hdr=0x%02X cmd=0x%02X cksum=0x%02X\n",
                      spi_rx_buf[0], spi_rx_buf[1], spi_rx_buf[2]);
    }
}

// ── FSM ──────────────────────────────────────────────────────────────────────
void handleEvent() {
    switch (currentState) {

        case State::IDLE:
            currentStatus = STATUS_READY;
            if (detected) {
                currentState  = State::MOVE;
                moveStartTime = millis();
                Serial.println("[FSM] IDLE -> MOVE");
            }
            break;

        case State::MOVE:
            currentStatus = STATUS_BUSY;
            if (millis() - moveStartTime >= 10000) {
                currentState = State::ERROR;
                Serial.println("[FSM] MOVE -> ERROR (timeout)");
            } else if (millis() - moveStartTime >= 1000) {
                currentState  = State::DROP;
                dropStartTime = millis();
                Serial.println("[FSM] MOVE -> DROP");
            }
            break;

        case State::DROP:
            currentStatus = STATUS_BUSY;
            if (millis() - dropStartTime >= 10000) {
                currentState = State::ERROR;
                Serial.println("[FSM] DROP -> ERROR (timeout)");
            } else if (millis() - dropStartTime >= 1000) {
                currentState = State::DONE;
                Serial.println("[FSM] DROP -> DONE");
            }
            break;

        case State::DONE:
            currentStatus = STATUS_DONE;
            delay(1000);
            detected     = false;
            currentState = State::IDLE;
            Serial.println("[FSM] DONE -> IDLE");
            break;

        case State::ERROR:
            currentStatus = STATUS_ERROR;
            break;
    }
}

// ── Servo driver ─────────────────────────────────────────────────────────────
/*
 * Servo 3 + 4 steer items into one of three bins:
 *
 * BIN_1 (left)   : servo3=15,  servo4=180   (WHITE / COTTON)
 * BIN_2 (centre) : servo3=90,  servo4=90    (DENIM)
 * BIN_3 (right)  : servo3=105, servo4=0     (NON_WHITE / FLEECE)
 *
 * Servo 1 + 2 drive the drop mechanism (unchanged).
 */

static State lastServoState = State::IDLE;

void driveServos() {
    if (currentState == lastServoState) return;
    lastServoState = currentState;

    switch (currentState) {

        case State::MOVE:
            switch (detectedBin) {
                case BinCmd::BIN_1:
                    servopos3 = 45;
                    servopos4 = 200;
                    break;
                case BinCmd::BIN_2:
                    servopos3 = 90;
                    servopos4 = 90;
                    break;
                case BinCmd::BIN_3:
                    servopos3 = 150;
                    servopos4 = 0;
                    break;
            }
            servoWrite(CH_SERVO4, servopos4);
            servoWrite(CH_SERVO3, servopos3);
            Serial.printf("[SERVO] Diverter -> BIN_%d  (s3=%d s4=%d)\n",
                          (int)detectedBin + 1, servopos3, servopos4);
            break;

        case State::DROP:
            servopos1 = 180;
            servopos2 = 270;
            servoWrite(CH_SERVO1, servopos1);
            servoWrite(CH_SERVO2, servopos2);
            delay(1000);
            servopos1 = 90;
            servopos2 = 90;
            servoWrite(CH_SERVO1, servopos1);
            servoWrite(CH_SERVO2, servopos2);
            break;

        case State::IDLE:
            servopos1 = 90;
            servopos2 = 90;
            servopos3 = 90;
            servopos4 = 90;
            servoWrite(CH_SERVO1, servopos1);
            servoWrite(CH_SERVO2, servopos2);
            servoWrite(CH_SERVO3, servopos3);
            servoWrite(CH_SERVO4, servopos4);
            break;

        case State::DONE:
            servopos1 = 90;
            servopos2 = 90;
            servopos3 = 90;
            servopos4 = 90;
            servoWrite(CH_SERVO1, servopos1);
            servoWrite(CH_SERVO2, servopos2);
            servoWrite(CH_SERVO3, servopos3);
            servoWrite(CH_SERVO4, servopos4);
            break;

        case State::ERROR:
            break;
    }
}

// ── Arduino entry points ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("[BOOT] ESP32-C6 SPI Slave + FSM + PWM starting...");
    setupLEDC();
    setupSPISlave();
    Serial.println("[BOOT] Ready.");
}

void loop() {
    pollSPI();
    handleEvent();
    driveServos();
    delay(10);
}