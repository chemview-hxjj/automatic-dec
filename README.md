# Automated Control System for Determining the Decomposition Equilibrium Constant of Ammonium Carbamate

This project is an automated control program for the experiment on determining the decomposition equilibrium constant of ammonium carbamate. The system uses a camera to monitor the liquid levels in a U-tube manometer in real time. In combination with a pressure sensor and solenoid valves, it automatically performs evacuation, equilibrium-state determination, and data recording. A web-based graphical interface is provided for convenient experimental operation and monitoring.

## Features

* **Real-time visual detection:** Uses OpenCV to identify the liquid-level positions of the red liquid in the U-tube manometer and supports automatic or manual setting of the bottom reference line.

* **Automatic control logic:** Automatically controls the opening and closing of solenoid valves according to the liquid-level difference, adjusts the pressure on both sides, and continues until equilibrium is reached.

* **Pressure monitoring:** Reads pressure-sensor data through a serial port for real-time display and vacuum control.

* **Data recording:** Automatically records experimental time, pressure, liquid-level state, and other data in CSV files and generates log files.

* **Web-based graphical interface:** Provides an intuitive cross-platform desktop application based on Flask and pywebview.

* **Configurable parameters:** HSV thresholds, liquid-level determination thresholds, vacuum-holding time, and other parameters can be adjusted through the interface or configuration files.

## Runtime Environment

* Operating system: Windows
* Python version: Python 3.11 or later

## Installation and Dependencies

1. Clone or download the project source code.

2. In the project root directory, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Program

Run the following command in the project root directory:

```bash
python main.py
```
