#!/usr/bin/env python3
"""
Serial Data Collector for MAX30102 Hemoglobin Prediction
========================================================

Collects RED and IR data from a COM port for 60 seconds, then uses the
existing csv_multi_predictor.py script to predict hemoglobin levels.

Expected serial data format:
114797,102863
115095,103599
115654,104888
...

Where first number is RED and second is IR.
"""

import serial
import time
import csv
import os
import sys
from datetime import datetime
from csv_multi_predictor import process_csv_readings, print_summary

def list_available_ports():
    """List all available COM ports"""
    import serial.tools.list_ports
    
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("❌ No COM ports found")
        return []
    
    print("\n📡 Available COM Ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device} - {port.description}")
    
    return ports

def collect_serial_data(port_name, baud_rate=9600, collection_time=60):
    """
    Collect RED and IR data from serial port for specified time
    
    Args:
        port_name (str): COM port name (e.g., 'COM3')
        baud_rate (int): Baud rate for serial communication
        collection_time (int): Time in seconds to collect data
        
    Returns:
        str: Path to saved CSV file or None if failed
    """
    
    print(f"🔌 Connecting to {port_name} at {baud_rate} baud...")
    
    try:
        # Open serial connection
        ser = serial.Serial(
            port=port_name,
            baudrate=baud_rate,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        
        # Wait for connection to stabilize
        time.sleep(2)
        print(f"✅ Connected to {port_name}")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"serial_data_{timestamp}.csv"
        
        # Prepare data collection
        red_values = []
        ir_values = []
        start_time = time.time()
        
        print(f"📊 Starting data collection for {collection_time} seconds...")
        print("Press Ctrl+C to stop early")
        
        # Create CSV file and write header
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Red', 'IR'])  # Header row
            
            reading_count = 0
            
            try:
                while (time.time() - start_time) < collection_time:
                    # Read line from serial
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line:
                        try:
                            # Parse RED,IR values
                            parts = line.split(',')
                            if len(parts) >= 2:
                                red_val = int(parts[0].strip())
                                ir_val = int(parts[1].strip())
                                
                                # Validate reasonable ranges
                                if 50000 <= red_val <= 200000 and 50000 <= ir_val <= 200000:
                                    red_values.append(red_val)
                                    ir_values.append(ir_val)
                                    writer.writerow([red_val, ir_val])
                                    reading_count += 1
                                    
                                    # Show progress every 10 readings
                                    if reading_count % 10 == 0:
                                        elapsed = time.time() - start_time
                                        remaining = collection_time - elapsed
                                        print(f"📈 Readings: {reading_count}, Time remaining: {remaining:.1f}s")
                                else:
                                    print(f"⚠️ Skipping invalid reading: {line}")
                        
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Error parsing line '{line}': {e}")
            
            except KeyboardInterrupt:
                print(f"\n🛑 Collection stopped by user")
        
        # Close serial connection
        ser.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Data collection complete!")
        print(f"⏱️ Collection time: {elapsed_time:.1f} seconds")
        print(f"📊 Total readings: {reading_count}")
        print(f"💾 Data saved to: {csv_filename}")
        
        if reading_count == 0:
            print("❌ No valid readings collected!")
            return None
        
        # Show sample of collected data
        if reading_count > 0:
            print(f"\n📋 Sample readings:")
            for i in range(min(5, len(red_values))):
                print(f"  {red_values[i]},{ir_values[i]}")
            if len(red_values) > 5:
                print(f"  ... and {len(red_values)-5} more")
        
        return csv_filename
        
    except serial.SerialException as e:
        print(f"❌ Serial communication error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def get_user_input():
    """Get user input for COM port, patient info, etc."""
    
    # List available ports
    ports = list_available_ports()
    if not ports:
        return None, None, None, None
    
    # Select COM port
    while True:
        try:
            choice = int(input(f"\nSelect COM port (1-{len(ports)}): ")) - 1
            if 0 <= choice < len(ports):
                port_name = ports[choice].device
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    # Get baud rate
    while True:
        baud_input = input("Enter baud rate (default 9600): ").strip()
        if not baud_input:
            baud_rate = 9600
            break
        try:
            baud_rate = int(baud_input)
            break
        except ValueError:
            print("Invalid baud rate!")
    
    # Get collection time
    while True:
        time_input = input("Collection time in seconds (default 60): ").strip()
        if not time_input:
            collection_time = 60
            break
        try:
            collection_time = int(time_input)
            if collection_time > 0:
                break
            else:
                print("Collection time must be positive!")
        except ValueError:
            print("Invalid time!")
    
    # Get patient information
    while True:
        gender = input("Enter gender (Male/Female): ").strip().capitalize()
        if gender in ['Male', 'Female']:
            break
        print("Please enter 'Male' or 'Female'")
    
    while True:
        try:
            age = int(input("Enter age: "))
            if age > 0:
                break
            else:
                print("Age must be positive!")
        except ValueError:
            print("Please enter a valid age!")
    
    return port_name, baud_rate, collection_time, gender, age

def main():
    """Main function"""
    print("🩸 MAX30102 SERIAL DATA COLLECTOR & HEMOGLOBIN PREDICTOR")
    print("=" * 60)
    print("This script will:")
    print("1. Collect RED and IR data from a COM port")
    print("2. Save the data to a CSV file")
    print("3. Use the data to predict hemoglobin levels")
    print("=" * 60)
    
    # Get user input
    user_input = get_user_input()
    if user_input[0] is None:
        return
    
    port_name, baud_rate, collection_time, gender, age = user_input
    
    print(f"\n🔧 Configuration:")
    print(f"COM Port: {port_name}")
    print(f"Baud Rate: {baud_rate}")
    print(f"Collection Time: {collection_time} seconds")
    print(f"Patient: {gender}, {age} years old")
    
    input("\nPress Enter to start data collection...")
    
    # Collect serial data
    csv_file = collect_serial_data(port_name, baud_rate, collection_time)
    
    if csv_file is None:
        print("❌ Data collection failed!")
        return
    
    print(f"\n🧠 Starting hemoglobin prediction using {csv_file}...")
    
    # Process the collected data using existing predictor
    try:
        results = process_csv_readings(csv_file, gender, age)
        
        if results is not None:
            # Print summary
            print_summary(results, gender, age)
            
            # Save detailed results
            output_file = f"serial_hemoglobin_results_{gender.lower()}_{age}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Serial Data Hemoglobin Prediction Results\n")
                f.write(f"="*50 + "\n")
                f.write(f"Data Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"COM Port: {port_name}\n")
                f.write(f"Collection Duration: {collection_time} seconds\n")
                f.write(f"CSV File: {csv_file}\n")
                f.write(f"Patient: {gender}, {age} years\n\n")
                f.write(f"Recommended Hemoglobin: {results['mean_hemoglobin']:.2f} g/dL\n")
                f.write(f"95% Confidence Interval: {results['confidence_95_lower']:.2f} - {results['confidence_95_upper']:.2f} g/dL\n")
                f.write(f"Standard Deviation: {results['std_deviation']:.3f} g/dL\n")
                f.write(f"Coefficient of Variation: {results['coefficient_variation']:.1f}%\n\n")
                f.write(f"All predictions: {results['all_predictions']}\n")
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
        else:
            print("❌ Hemoglobin prediction failed!")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("Please check that the hemoglobin_svr_model.pkl file exists in the current directory")

if __name__ == "__main__":
    # Check if required modules are available
    try:
        import serial
        import serial.tools.list_ports
    except ImportError:
        print("❌ PySerial not found!")
        print("Please install it using: pip install pyserial")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists('hemoglobin_svr_model.pkl'):
        print("❌ Model file 'hemoglobin_svr_model.pkl' not found!")
        print("Please ensure the model file is in the current directory")
        sys.exit(1)
    
    main()