# DFRobot_VEML7700
* [中文版](./README_CN.md)

VEML7700 is a high accuracy ambient light digital 16-bit resolution sensor in a mininature transparent 6.8mm×2.35mm×3.0mm package.
It includes a high sensitive photo diode, a low noise amplifier, a 16-bit A/D converter and supports an easy-to-use Gravity I2C bus communication interface.
The ambient light result is available as digital value.

![产品实物图](../../resources/images/VEML7700.png)


## Product Link (https://www.dfrobot.com/product-1620.html)
    SKU：SEN0228


## Table of Contents

* [Summary](#summary)
* [Installation](#installation)
* [Methods](#methods)
* [Compatibility](#compatibility)
* [History](#history)
* [Credits](#credits)


## Summary

* Filtrate tm technology adoption: close to real human eye reaction
* O-TrimTM technology adoption: ALS output tolerance
≤10%
* 16-bit dynamic range for ambient light detection from 0 lx to 120 KLX with resolution 0.0036 lx/ct, support low transmittance (dark) lens design
* 100hz and 120hz flicker noise rejection
* Excellent temperature compensation
* High dynamic detection resolution
* Software shutdown mode control


## Installation

Download the library file before use, paste it into the specified directory, then open the Examples folder and run the demo in the folder.


## Methods

```python

    '''!
      @brief Init function
      @return  Return init status
      @retval True indicate init succeed
      @retval False indicate init failed
    '''
    def begin(self):

    '''!
      @brief Get measured ALS illumination intensity value
      @return Return the processed ALS illumination intensity value
    '''
    def get_ALS_lux(self):

    '''!
      @brief Get measured White illumination intensity value
      @return Return the processed White illumination intensity value
    '''
    def get_white_lux(self):

    '''!
      @brief Set ALS gain coefficient
      @param gain Gain selection:
      @n       ALS_GAIN_x1, ALS_GAIN_x2, ALS_GAIN_d8, ALS_GAIN_d4
    '''
    def set_gain(self, gain):

    '''!
      @brief Get ALS gain coefficient
      @return The current set ALS gain coefficient:
      @n        ALS_GAIN_x1, ALS_GAIN_x2, ALS_GAIN_d8, ALS_GAIN_d4
    '''
    def get_gain(self):

    '''!
      @brief Set ALS integration time
      @param itime ALS integration time setting:
      @n       ALS_INTEGRATION_25ms, ALS_INTEGRATION_50ms, ALS_INTEGRATION_100ms, 
      @n       ALS_INTEGRATION_200ms, ALS_INTEGRATION_400ms, ALS_INTEGRATION_800ms
    '''
    def set_integration_time(self, itime):

    '''!
      @brief Get ALS integration time
      @return The current set ALS integration time:
      @n       ALS_INTEGRATION_25ms, ALS_INTEGRATION_50ms, ALS_INTEGRATION_100ms, 
      @n       ALS_INTEGRATION_200ms, ALS_INTEGRATION_400ms, ALS_INTEGRATION_800ms
    '''
    def get_integration_time(self):

    '''!
      @brief persistence protect number setting
      @param persist persistence protect number setting:
      @n       ALS_PERSISTENCE_1, ALS_PERSISTENCE_2, ALS_PERSISTENCE_4, ALS_PERSISTENCE_8
    '''
    def set_persistence(self, persist):

    '''!
      @brief Power saving mode setting
      @param powerMode Power saving mode:
      @n       ALS_POWER_MODE_1, ALS_POWER_MODE_2, ALS_POWER_MODE_3, ALS_POWER_MODE_4
    '''
    def set_power_saving_mode(self, power_mode):

    '''!
      @brief Power saving mode enable
      @param enabled :
      @n       0 Disable
      @n       1 Enable
    '''
    def set_power_saving(self, enabled):

    """!
      @brief ALS interrupt enable setting
      @param enabled :
      @n       0 Disable
      @n       1 Enable
    """
    def set_interrupts(self, enabled):

    """!
      @brief ALS shut down setting
      @param on :
      @n       0 power on
      @n       1 shut down
    """
    def set_power(self, on):

    """!
      @brief Set high threshold of interrupt
      @param thresh 16-bit high threshold set value
    """
    def set_ALS_high_threshold(self, thresh):

    """!
      @brief Set low threshold of interrupt
      @param thresh 16-bit low threshold set value
    """
    def set_ALS_low_threshold(self, thresh):

        """!
          @brief Get high threshold trigger interrupt event
          @return High threshold trigger interrupt event mark :
          @retval   0 Not detect high threshold interrupt event
          @retval   1 Detected high threshold interrupt event
        """
    def get_high_threshold_event(self):

    """!
      @brief Get low threshold trigger interrupt event
      @return Low threshold trigger interrupt event mark :
      @retval   0 Not detect low threshold interrupt event
      @retval   1 Detected low threshold interrupt event
    """
    def get_low_threshold_event(self):

```

## Compatibility

* RaspberryPi version

| Board        | Work Well | Work Wrong | Untested | Remarks |
| ------------ | :-------: | :--------: | :------: | ------- |
| RaspberryPi2 |           |            |    √     |         |
| RaspberryPi3 |           |            |    √     |         |
| RaspberryPi4 |     √     |            |          |         |

* Python version

| Python  | Work Well | Work Wrong | Untested | Remarks |
| ------- | :-------: | :--------: | :------: | ------- |
| Python2 |     √     |            |          |         |
| Python3 |     √     |            |          |         |


## History

- 2021/10/15 - Version 1.0.0 released.


## Credits

Written by qsjhyy(yihuan.huang@dfrobot.com), 2021. (Welcome to our [website](https://www.dfrobot.com/))
