import tkinter as tk
from tkinter import ttk, messagebox
import subprocess #required for memory stress test subprocess
import platform
import re
import time
import math
import cpuinfo # This may require 'pip install py-cpuinfo'
import time
import psutil  # Install with: pip install psutil
import os # For file operations
import multiprocessing  # For CPU core count
import sys

try:
    import cpuinfo  # Requires 'pip install py-cpuinfo'
except ImportError:
    messagebox.showerror("Error", "Missing required package: py-cpuinfo\nPlease install it with 'pip install py-cpuinfo'")
    sys.exit(1)

"""
Next, for the AVX operations, there are a few approaches. 
The most straightforward would be to use NumPy, 
which provides vectorized operations that can utilize 
AVX under the hood. Another approach would be to use a 
library like PyVectorize or numba. But since the code is 
using very specific AVX/AVX-512 intrinsics like _mm256_set1_pd, 
it might be trying to simulate the exact Intel intrinsics.

For this purpose, I'll create simple Python implementations 
of these functions that mimic the behavior but note that 
this is not the same as using actual AVX/AVX-512 instructions. 
The proper way would be to use a library that binds to the 
actual Intel intrinsics (like PyVectorize) or use Numba's 
vectorize.

The AVX and AVX-512 simulations I've provided are just 
functional equivalents in Python - they don't actually use 
AVX/AVX-512 instructions which would provide hardware 
acceleration. For real hardware-accelerated tests, you 
would want to use C/C++ with intrinsics or a library 
that provides bindings to these intrinsics.

Here's how we handle the AVX/AVX-512 functions:
"""

# AVX/AVX-512 simulation (not actual hardware acceleration)
def _mm256_set1_pd(value):
    """Simulates _mm256_set1_pd which sets all elements to the same value"""
    return [value] * 4  # 4 doubles in AVX

def _mm256_add_pd(a, b):
    """Simulates _mm256_add_pd which adds packed double-precision values"""
    return [a[i] + b[i] for i in range(4)]

def _mm256_mul_pd(a, b):
    """Simulates _mm256_mul_pd which multiplies packed double-precision values"""
    return [a[i] * b[i] for i in range(4)]

def _mm256_div_pd(a, b):
    """Simulates _mm256_div_pd which divides packed double-precision values"""
    return [a[i] / b[i] for i in range(4)]

def _mm256_extract_f64(a, idx):
    """Simulates extracting a double-precision value from an AVX register"""
    return a[idx]

# AVX-512 simulation
def _mm512_set1_pd(value):
    """Simulates _mm512_set1_pd which sets all elements to the same value"""
    return [value] * 8  # 8 doubles in AVX-512

def _mm512_add_pd(a, b):
    """Simulates _mm512_add_pd which adds packed double-precision values"""
    return [a[i] + b[i] for i in range(8)]

def _mm512_mul_pd(a, b):
    """Simulates _mm512_mul_pd which multiplies packed double-precision values"""
    return [a[i] * b[i] for i in range(8)]

def _mm512_div_pd(a, b):
    """Simulates _mm512_div_pd which divides packed double-precision values"""
    return [a[i] / b[i] for i in range(8)]

def _mm512_extract_f64(a, idx):
    """Simulates extracting a double-precision value from an AVX-512 register"""
    return a[idx]


### Begin Main Program and GUI ###

class ProcessorTesterApp:
    def __init__(self, master):
        self.master = master
        master.title("Intel Processor Tester")
        master.geometry("1000x600")  # Larger window
        master.configure(bg="white")  # White background
        master.resizable(False, False)

        # --- Main Frame ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.configure(style='White.TFrame')

        # --- Style for white background ---
        style = ttk.Style()
        style.configure('White.TFrame', background='white')
        style.configure('White.TLabel', background='white')
        style.configure('White.TLabelframe', background='white')
        style.configure('White.TLabelframe.Label', background='white')


        # --- CPU Information Frame (Left Side) ---
        self.cpu_info_frame = ttk.LabelFrame(self.main_frame, text="CPU Information", padding="10")
        self.cpu_info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.cpu_info_frame.configure(style='White.TLabelframe')

        self.cpu_info_text = tk.Text(self.cpu_info_frame, wrap=tk.WORD, bg="white", bd=0, height=20, width=40)
        self.cpu_info_text.pack(fill=tk.BOTH, expand=True)
        self.display_cpu_info()


        # --- Test Modules Frame (Right Side) ---
        self.test_modules_frame = ttk.LabelFrame(self.main_frame, text="Test Modules", padding="10")
        self.test_modules_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.test_modules_frame.configure(style='White.TLabelframe')

        # --- Define Test Modules ---
        self.test_modules = [
            "Genuine Intel", "BrandString", "Cache", "IMMXSSE",
            "IMC", "Prime Number", "Floating Point", "Math"
        ]
        self.test_buttons = {}
        self.result_labels = {}

        # --- Create Buttons and Result Labels ---
        row_num = 0
        for module in self.test_modules:
            button = ttk.Button(self.test_modules_frame, text=module,
                                command=lambda m=module: self.run_test(m))
            button.grid(row=row_num, column=0, sticky=tk.W, padx=5, pady=5)
            self.test_buttons[module] = button

            result_label = ttk.Label(self.test_modules_frame, text="Not Tested", width=15)
            result_label.grid(row=row_num, column=1, sticky=tk.W, padx=5, pady=5)
            result_label.configure(style='White.TLabel')
            self.result_labels[module] = result_label

            row_num += 1

        # --- Overall Status ---
        self.status_label = ttk.Label(self.main_frame, text="Overall Status:  Pending", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=10)
        self.status_label.configure(style='White.TLabel')

        # --- Run All Tests Button ---
        self.run_all_button = ttk.Button(self.main_frame, text="Run All Tests", command=self.run_all_tests)
        self.run_all_button.pack(pady=5)

        # --- IMC Test Variables ---
        self.arg_exp_memory = 0
        self.arg_memory_size_tolerance = 0
        self.skip_memory_size_test = 0
        self.skip_memory_stress_test = 0  # Added stress test skip
        self.f_arg_memory_size_tolerance = 0.10  # Default tolerance (10%)
        self.s_arg_exp_memory = ""
        self.display_only = 0  # -nc option
        self.imc_stress_test_process = None # To store the stress test subprocess

        # --- Prime Number Test Variables ---
        self.prime_number_timer = 2  # Default 2 seconds
        self.stop_on_error = False
        self.display_only_prime = False
        self.avx_level = 0  # 0=No AVX, 1=AVX, 2=AVX2, 3=AVX512

        # --- Floating Point Test Variables ---
        self.fp_timer = 5  # Default to 5 seconds, same as fma_timer
        self.stop_on_error_fp = False # Default to continuing on error
        self.display_only_fp = False # Default to comparing results

        # --- FMA3/Math Test Variables ---
        self.fma_timer = 5 # Default to 5 seconds

        # Create log directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)


    def display_cpu_info(self):
        """Displays detailed CPU information in the text area."""
        try:
            cpu_info_dict = cpuinfo.get_cpu_info()
            num_processors = multiprocessing.cpu_count()  # Total logical processors
            physical_cores = psutil.cpu_count(logical=False)  # Physical cores

            info_str = f"Processor Name: {cpu_info_dict.get('brand_raw', 'N/A')}\n"
            info_str += f"Architecture: {cpu_info_dict.get('arch_string_raw', 'N/A')}\n"
            info_str += f"Number of Processors (Logical): {num_processors}\n"
            info_str += f"Number of Physical Cores: {physical_cores}\n"
            info_str += f"Vendor ID: {cpu_info_dict.get('vendor_id_raw', 'N/A')}\n"
            info_str += f"L1 Cache: {cpu_info_dict.get('l1_data_cache', 'N/A')}\n"
            info_str += f"L2 Cache: {cpu_info_dict.get('l2_cache_size', 'N/A')}\n"
            info_str += f"L3 Cache: {cpu_info_dict.get('l3_cache_size', 'N/A')}\n"


            # Add more fields as needed
            self.cpu_info_text.insert(tk.END, info_str)
            self.cpu_info_text.config(state=tk.DISABLED) # Make read-only

        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve CPU information:\n{e}")

    def run_test(self, module_name):
        """Runs a single test based on the module name."""

        if module_name == "Genuine Intel":
            self.run_genuine_intel_test()
        elif module_name == "BrandString":
            self.run_brand_string_test()
        elif module_name == "Cache":
            self.run_cache_test()
        elif module_name == "IMMXSSE":
            self.run_immxsse_test()
        elif module_name == "IMC":
            self.run_imc_test()
        elif module_name == "Prime Number":
            self.run_prime_number_test()
        elif module_name == "Floating Point":
            self.run_floating_point_test()
        elif module_name == "Math":
            self.run_math_fma_test()
        else:
            messagebox.showerror("Error", f"Test module '{module_name}' not ready.")
            self.result_labels[module_name].config(text="Error", foreground="red")
            self.update_overall_status()

        self.update_overall_status()

    def run_genuine_intel_test(self):
        """Performs the Genuine Intel test."""
        try:
            cpu_info = cpuinfo.get_cpu_info()
            vendor_id = cpu_info.get('vendor_id_raw', 'Unknown')

            if vendor_id == "GenuineIntel":
                result = "PASS"
                color = "green"
            else:
                result = "FAIL"
                color = "red"

            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                vendor_id = f"{vendor_id} (OS Not Supported)"

            self.result_labels["Genuine Intel"].config(text=result, foreground=color)
            self.write_log_file("Genuine Intel", result, vendor_id)

        except Exception as e:
            self.result_labels["Genuine Intel"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"Genuine Intel Test Error: {e}")
            self.write_log_file("Genuine Intel", "ERROR", str(e))


    def run_brand_string_test(self):
        """Performs the Brand String test and displays the result."""
        try:
            cpu_info = cpuinfo.get_cpu_info()
            brand_string = cpu_info.get('brand_raw', 'Unknown')

            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                brand_string = f"{brand_string} (OS Not Supported)"
                self.result_labels["BrandString"].config(text=result, foreground=color)
                self.write_log_file("BrandString", result, brand_string)
                return

            # Check for '#' in brand string (early sample) - Keep this check
            if "#" in brand_string:
                result = "WARNING"  # Change to WARNING
                color = "orange"    # Use orange for warnings
                details = "Possible Early Engineering Sample Processor Detected"
                self.result_labels["BrandString"].config(text=result, foreground=color)
                self.write_log_file("BrandString", result, f"{brand_string} - {details}")
                return #Keep going to display the brand string

            # Display the brand string directly
            result = "DETECTED"  # Neutral status
            color = "blue"       # Blue for information
            self.result_labels["BrandString"].config(text=result, foreground=color)
            self.write_log_file("BrandString", "INFO", brand_string)


        except Exception as e:
            self.result_labels["BrandString"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"Brand String Test Error: {e}")
            self.write_log_file("BrandString", "ERROR", str(e))

    def run_cache_test(self):
        """Performs comprehensive cache performance and stability tests using Intel MLC."""
        try:
            # --- 1. OS and MLC Path Check ---
            if platform.system() != "Windows":  # MLC is primarily a Windows tool
                self.result_labels["Cache"].config(text="OS Not Supported", foreground="red")
                messagebox.showerror("Cache Test", "This test requires Windows and Intel MLC.")
                self.write_log_file("Cache", "OS Not Supported", "Requires Windows")
                return

            mlc_path = "path/to/mlc.exe"  # *** REPLACE WITH THE ACTUAL PATH TO mlc.exe ***
            if not os.path.exists(mlc_path):
                self.result_labels["Cache"].config(text="MLC Not Found", foreground="red")
                messagebox.showerror("Cache Test", "Intel MLC executable not found at the specified path.")
                self.write_log_file("Cache", "MLC Not Found", f"Expected at: {mlc_path}")
                return

            # --- 2. Run MLC Tests ---
            # Run bandwidth and latency tests.  Adjust --time as needed.
            # --bandwidth_matrix gives bandwidth in GB/s
            # --latency_matrix gives latency in nanoseconds.
            # -t<seconds>: Specifies test duration.  Longer is better for stability.
            # -e: exclude specified tests

            # Example 1: Basic Bandwidth and Latency Test (Short)
            test_duration = 10 # 10 seconds.  Increase for a *real* stress test (e.g., 600 for 10 minutes)
            result = subprocess.run(
                [mlc_path, "--bandwidth_matrix", "--latency_matrix", f"-t{test_duration}"],
                capture_output=True,
                text=True,
                check=True,  # Raise exception on non-zero exit code
                timeout=test_duration + 30 # Add a buffer to the timeout
            )
            
            output = result.stdout
            # --- 3. Parse MLC Output (CRITICAL: Adapt to the actual output) ---

            results = {}  # Store parsed results

            # --- Bandwidth Parsing (Example - Adapt to MLC's output format) ---
            # Look for lines like: "  Max   :  23.45   34.56   12.34"  (numbers are GB/s)
            for line in output.splitlines():
                if "Max   :" in line:
                    parts = line.split(":")
                    bandwidth_values = [float(x) for x in parts[1].split()]
                    #  You might have multiple bandwidth values (for different cores/sockets).
                    #  Decide how you want to handle them (e.g., average, max, min).
                    results['Max Bandwidth'] = max(bandwidth_values)  # Example: Store the maximum bandwidth

            # --- Latency Parsing (Example - Adapt to MLC's output format) ---
            #  Look for lines related to different cache levels (L1, L2, L3, Memory).
            #  The exact format will depend on the MLC version.
            for level in ["L1", "L2", "L3", "Memory"]:
              for line in output.splitlines():
                if f"{level} latency" in line:
                    #extract number with regular expression.
                    match = re.search(r"(\d+\.?\d*)", line)
                    if match:
                         results[f'{level} Latency (ns)'] = float(match.group(1))
                         break #stop searching once we have the result.

            # --- 4. Analyze Results and Set Status ---

            passed = True  # Assume pass initially
            log_details = ""

            # --- Example Thresholds (ADJUST THESE BASED ON YOUR CPU AND EXPECTATIONS) ---
            bandwidth_threshold = 20  # GB/s (Example - Adjust based on your CPU)
            l1_latency_threshold = 2  # ns
            l2_latency_threshold = 10  # ns
            l3_latency_threshold = 40  # ns
            memory_latency_threshold = 100 # ns

            if 'Max Bandwidth' in results:
                log_details += f"Max Bandwidth: {results['Max Bandwidth']:.2f} GB/s\n"
                if results['Max Bandwidth'] < bandwidth_threshold:
                    passed = False
                    log_details += f"  WARNING: Bandwidth below threshold ({bandwidth_threshold} GB/s)\n"
            else:
                passed = False
                log_details += "  ERROR: Could not parse Max Bandwidth\n"

            for level in ["L1", "L2", "L3", "Memory"]:
                if f'{level} Latency (ns)' in results:
                    latency = results[f'{level} Latency (ns)']
                    log_details += f"{level} Latency: {latency:.2f} ns\n"
                    if level == "L1" and latency > l1_latency_threshold:
                        passed = False;
                        log_details += f"  WARNING: {level} Latency above threshold ({l1_latency_threshold} ns)\n"
                    if level == "L2" and latency > l2_latency_threshold:
                        passed = False;
                        log_details += f"  WARNING: {level} Latency above threshold ({l2_latency_threshold} ns)\n"
                    if level == "L3" and latency > l3_latency_threshold:
                        passed = False;
                        log_details += f"  WARNING: {level} Latency above threshold ({l3_latency_threshold} ns)\n"
                    if level == "Memory" and latency > memory_latency_threshold:
                        passed = False;
                        log_details += f"  WARNING: {level} Latency above threshold ({memory_latency_threshold} ns)\n"

                else:
                    passed = False
                    log_details += f"  ERROR: Could not parse {level} Latency\n"


            # --- 5. Update GUI and Log File ---
            if passed:
                self.result_labels["Cache"].config(text="Pass", foreground="green")
            else:
                self.result_labels["Cache"].config(text="Fail", foreground="red")

            self.write_log_file("Cache", "Pass" if passed else "Fail", log_details)

        except subprocess.CalledProcessError as e:
            self.result_labels["Cache"].config(text="Fail", foreground="red")
            messagebox.showerror("Cache Test", f"MLC returned an error:\n{e.stderr}")
            self.write_log_file("Cache", "Fail", f"MLC Error: {e.stderr}")
        except subprocess.TimeoutExpired:
            self.result_labels["Cache"].config(text="Fail", foreground="red")
            messagebox.showerror("Cache Test", "MLC test timed out.")
            self.write_log_file("Cache", "Fail", "MLC Timeout")
        except FileNotFoundError:
            self.result_labels["Cache"].config(text="Fail", foreground="red")
            messagebox.showerror("Cache Test", "MLC executable not found.")
            self.write_log_file("Cache", "Fail", "MLC Not Found")
        except Exception as e:
            self.result_labels["Cache"].config(text="ERROR", foreground="red")
            messagebox.showerror("Cache Test", f"An unexpected error occurred: {e}")
            self.write_log_file("Cache", "ERROR", str(e))

    def run_immxsse_test(self):
        """Performs the IMMX/SSE instruction set verification."""
        try:
            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                details = "OS Not Supported"
                self.result_labels["IMMXSSE"].config(text=result, foreground=color)
                self.write_log_file("IMMXSSE", result, details)
                return

            # Get CPU flags
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])
            
            # Check for MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2
            mmx = 'mmx' in flags
            sse = 'sse' in flags
            sse2 = 'sse2' in flags
            sse3 = 'sse3' in flags or 'pni' in flags  # pni is another name for SSE3
            ssse3 = 'ssse3' in flags
            sse4_1 = 'sse4_1' in flags
            sse4_2 = 'sse4_2' in flags
            
            # Build result details
            details = []
            details.append(f"MMX: {'Yes' if mmx else 'No'}")
            details.append(f"SSE: {'Yes' if sse else 'No'}")
            details.append(f"SSE2: {'Yes' if sse2 else 'No'}")
            details.append(f"SSE3: {'Yes' if sse3 else 'No'}")
            details.append(f"SSSE3: {'Yes' if ssse3 else 'No'}")
            details.append(f"SSE4.1: {'Yes' if sse4_1 else 'No'}")
            details.append(f"SSE4.2: {'Yes' if sse4_2 else 'No'}")
            
            # Determine result
            if mmx and sse and sse2:  # Basic requirement
                result = "PASS"
                color = "green"
            else:
                result = "FAIL"
                color = "red"
                
            self.result_labels["IMMXSSE"].config(text=result, foreground=color)
            self.write_log_file("IMMXSSE", result, "\n".join(details))
            
        except Exception as e:
            self.result_labels["IMMXSSE"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"IMMXSSE Test Error: {e}")
            self.write_log_file("IMMXSSE", "ERROR", str(e))

    def run_imc_test(self):
        """Performs the IMC test (memory size check)."""
        try:
            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                details = "OS Not Supported"
                self.result_labels["IMC"].config(text=result, foreground=color)
                self.write_log_file("IMC", result, details)
                return
            # --- Check if IMC is supported (using CPUID) ---
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])
            model = cpu_info.get('model')
            family = cpu_info.get('family')
            stepping = cpu_info.get('stepping')

            #The IMC test is only supported in certain processors.
            if ((family == 0x6) and (model == 0x1D)) or ((family == 0x6) and (model == 0x17)) or ((family == 0x6) and (model == 0x0F)) or ((family == 0x6) and (model == 0x0E)) or ((family == 0x6) and (model == 0x0D)) or ((family == 0x6) and (model == 0x06)) or ((family == 0x6) and (model == 0x03)) or ((family == 0x6) and (model == 0x04)) or ((family == 0x6) and (model == 0x09)):
                result = "NOT SUPPORTED"
                color = "orange"
                details = "IMC Test Not Supported on this Processor"
                self.result_labels["IMC"].config(text=result, foreground=color)
                self.write_log_file("IMC", result, details)
                return
            
            # --- Get Total Physical Memory (using psutil) ---
            mem = psutil.virtual_memory()
            total_memory_bytes = mem.total
            total_memory_gb = total_memory_bytes / (1024**3)  # Convert to GB

            # --- Handle -nc option (display only) ---
            if self.display_only:
                result = "NO COMPARE"
                color = "blue"
                details = f"Detected Memory: {total_memory_gb:.2f} GB (No Comparison)"
                self.result_labels["IMC"].config(text=result, foreground=color)
                self.write_log_file("IMC", result, details)
                return
            
            # --- Skip Memory Size Test if Requested---
            if self.skip_memory_size_test:
                result = "DETECTED"
                color = "blue"
                details = f"Detected Memory: {total_memory_gb:.2f} GB (Size Test Skipped)"
                self.result_labels["IMC"].config(text=result, foreground=color)
                self.write_log_file("IMC", result, details)
                #Do not return to allow stress test to continue
            
            # --- Memory Size Comparison (if not skipped and expected size provided) ---
            if not self.skip_memory_size_test and self.s_arg_exp_memory:
                try:
                    expected_memory_gb = self.parse_memory_size(self.s_arg_exp_memory)
                    tolerance = self.f_arg_memory_size_tolerance

                    lower_bound = expected_memory_gb * (1 - tolerance)
                    upper_bound = expected_memory_gb * (1 + tolerance)

                    if lower_bound <= total_memory_gb <= upper_bound:
                        result = "PASS"
                        color = "green"
                        details = (f"Detected Memory: {total_memory_gb:.2f} GB, "
                                   f"Expected: {expected_memory_gb:.2f} GB (Tolerance: {tolerance*100:.2f}%)")
                    else:
                        result = "FAIL"
                        color = "red"
                        details = (f"Detected Memory: {total_memory_gb:.2f} GB, "
                                   f"Expected: {expected_memory_gb:.2f} GB (Tolerance: {tolerance*100:.2f}%) - Out of Range")
                except ValueError:
                    result = "ERROR"
                    color = "red"
                    details = "Invalid expected memory size format."

                self.result_labels["IMC"].config(text=result, foreground=color)
                self.write_log_file("IMC", result, details)

            elif not self.skip_memory_size_test: #If not skipped, and no args provided
              result = "DETECTED"
              color = "blue"
              details = f"Detected Memory: {total_memory_gb:.2f} GB"
              self.result_labels["IMC"].config(text=result, foreground=color)
              self.write_log_file("IMC", result, details)

            #--- Skip Memory Stress Test
            if self.skip_memory_stress_test:
                return #Skip stress test.

            # --- Memory Stress Test ---
            # if result == "PASS" or result == "DETECTED":  # Only run stress test if size check passed (or was skipped)
            # For simplicity, and to avoid complex memory management in Python,
            # let's just allocate a small amount of memory and hold it.
            # A *real* stress test would need to be a separate process.

            #   messagebox.showinfo("IMC Stress Test", "Running IMC Stress Test. This might take a while...")
            #   # This is a placeholder.  A real stress test is complex.
            #   time.sleep(5)  # Simulate some work
            #   messagebox.showinfo("IMC Stress Test", "IMC Stress Test Complete.")
            #   # We don't change the result based on this placeholder.
            self.run_memory_stress_test()


        except Exception as e:
            self.result_labels["IMC"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"IMC Test Error: {e}")
            self.write_log_file("IMC", "ERROR", str(e))

    def run_memory_stress_test(self):
        """Runs the IMC memory stress test (using compiled stress_test binary)"""
        try:
            # Build the command to execute
            command = ["./stress_test"]

            #Start the stress test process
            self.imc_stress_test_process = subprocess.Popen(command,
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE)

            # --- Update GUI to show stress test has started
            self.result_labels["IMC"].config(text="STRESS TEST RUNNING", foreground="orange")
            self.master.update() #Update the GUI to show the change

            # --- Wait for process to complete and get output
            stdout, stderr = self.imc_stress_test_process.communicate()
            return_code = self.imc_stress_test_process.returncode

            if return_code == 0:
              result = "PASS"
              color = "green"
              details = "IMC Stress Test Passed:\n" + stdout.decode()
            else:
              result = "FAIL"
              color = "red"
              details = "IMC Stress Test Failed:\n" + stdout.decode() + "\n" + stderr.decode()

            self.result_labels["IMC"].config(text=result, foreground=color)
            self.write_log_file("IMC-StressTest", result, details)

        except FileNotFoundError:
          messagebox.showerror("Error", "stress_test executable not found.  Make sure it's compiled and in the same directory.")
          self.result_labels["IMC"].config(text="ERROR", foreground="red")
          self.write_log_file("IMC-StressTest", "ERROR", "stress_test executable not found.")
        except Exception as e:
            self.result_labels["IMC"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"IMC Stress Test Error: {e}")
            self.write_log_file("IMC-StressTest", "ERROR", str(e))

    def parse_memory_size(self, size_str):
        """Parses a memory size string (e.g., '4GB', '2048MB') and returns the size in GB."""
        size_str = size_str.upper()
        match = re.match(r"([\d.]+)\s*([KMGT]?B)", size_str)
        if not match:
            raise ValueError("Invalid memory size format.")

        value, unit = match.groups()
        value = float(value)

        if unit == "KB":
            value /= 1024 * 1024
        elif unit == "MB":
            value /= 1024
        elif unit == "TB":
            value *= 1024
        return value

    def run_prime_number_test(self):
        """Performs the Prime Number Generation test."""
        try:
            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                details = "OS Not Supported"
                self.result_labels["Prime Number"].config(text=result, foreground=color)
                self.write_log_file("Prime Number", result, details)
                return

            # --- Get CPU Information ---
            cpu_info = cpuinfo.get_cpu_info()
            avx_flags = cpu_info.get('flags', [])  #flags provide AVX support information

            #Determine Max AVX Support
            avx_support_level = 0 #Default value
            if 'avx512f' in avx_flags:
                avx_support_level = 3
            elif 'avx2' in avx_flags:
                avx_support_level = 2
            elif 'avx' in avx_flags:
                avx_support_level = 1
            
            # --- Check if user specified AVX level is supported
            if self.avx_level > 0 and self.avx_level > avx_support_level:
                messagebox.showwarning("Warning", f"Selected AVX level ({self.avx_level}) is not supported. Using max supported level ({avx_support_level}).")
                self.avx_level = avx_support_level  # Use max supported
            
            # --- Run the Prime Number Test ---
            ops_per_sec, error_count = self.run_prime_number_calculation(self.prime_number_timer, self.avx_level)

            # --- Determine Result and Update Label ---
            if self.display_only_prime:
                result = "NO COMPARE"
                color = "blue"
                details = (f"Operations Per Second: {ops_per_sec}, "
                           f"Errors: {error_count} (No Comparison)")
            elif error_count == 0:
                result = "PASS"
                color = "green"
                details = f"Operations Per Second: {ops_per_sec}, Errors: {error_count}"
            else:
                result = "FAIL"
                color = "red"
                details = f"Operations Per Second: {ops_per_sec}, Errors: {error_count}"

            self.result_labels["Prime Number"].config(text=result, foreground=color)
            self.write_log_file("Prime Number", result, details)

        except Exception as e:
            self.result_labels["Prime Number"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"Prime Number Test Error: {e}")
            self.write_log_file("Prime Number", "ERROR", str(e))

    def is_prime(self, n):
        """Efficiently checks if a number is prime."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def run_prime_number_calculation(self, duration, avx_level=0):
        """Runs the prime number generation test for the specified duration.
           Uses optimized methods and AVX instructions if available.
        """
        start_time = time.time()
        end_time = start_time + duration
        ops_count = 0
        error_count = 0
        primes_found = 0 #Keep track of primes

        if avx_level == 3:
            ops_count, error_count, primes_found = self.prime_number_avx512(duration)
        elif avx_level == 2:
            ops_count, error_count, primes_found = self.prime_number_avx2(duration)
        elif avx_level == 1:
            ops_count, error_count, primes_found = self.prime_number_avx(duration)
        else: #No AVX
          while time.time() < end_time:
            # --- Basic Prime Number Generation (Optimized) ---
            for num in range(2, 10001):  # Check numbers up to 10000 (adjust as needed)
                if self.is_prime(num):
                  primes_found +=1
                ops_count += 1  # Count each number checked as an operation
            # --- End Basic Prime Number Generation ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        #Avoid division by cero error.
        ops_per_sec = ops_count / elapsed_time if elapsed_time > 0 else 0
        return int(ops_per_sec), error_count
    
    def prime_number_avx(self, duration):
        """Calculates prime numbers with AVX"""
        #Find out number of cores.
        num_cores = multiprocessing.cpu_count()
        #Variable to keep track of counts across threads.
        ops_count = 0
        error_count = 0
        primes_found = 0

        #Create a pool of worker processes.
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Divide work among processes. Each process gets a range.
            results = pool.starmap(self._prime_number_avx_range, [(duration, core_id) for core_id in range(num_cores)])
            #Combine the results from all the processes
            for result in results:
                ops_count += result[0]
                error_count += result[1]
                primes_found += result[2]
        return ops_count, error_count, primes_found

    def _prime_number_avx_range(self, duration, core_id):
        """Calculates prime numbers within a range using AVX instructions"""
        start_time = time.time()
        end_time = start_time + duration
        ops_count = 0
        error_count = 0
        primes_found = 0 #Keep track of primes

        # Ensure the ranges don't overlap
        range_start = 2 + core_id * 10000  # Example: Each core gets a range of 10000
        range_end = range_start + 10000

        while time.time() < end_time:
            for num in range(range_start, range_end):
                # Create an AVX register with 8 copies of the number
                num_vec = _mm256_set1_epi32(num)
                is_prime_avx = True

                # Check divisibility up to sqrt(num)
                for i in range(2, int(num**0.5) + 1):
                    divisor_vec = _mm256_set1_epi32(i)
                    # Perform division
                    quotient_vec = _mm256_div_epi32(num_vec, divisor_vec)
                    # Multiply back to check for remainder
                    remainder_vec = _mm256_mullo_epi32(quotient_vec, divisor_vec)
                    # Check for zero remainder
                    comparison_result = _mm256_cmpeq_epi32(remainder_vec, _mm256_setzero_si256())
                    # If any remainder is zero, it's not prime
                    if _mm256_movemask_epi8(comparison_result) != 0:
                        is_prime_avx = False
                        break
                if is_prime_avx:
                    primes_found += 1
                ops_count += 1 #Count loop iterations as operations
        return ops_count, error_count, primes_found

    def prime_number_avx2(self, duration):
        """Calculates prime numbers with AVX2"""
        #Find out number of cores.
        num_cores = multiprocessing.cpu_count()
        #Variable to keep track of counts across threads.
        ops_count = 0
        error_count = 0
        primes_found = 0

        #Create a pool of worker processes.
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Divide work among processes. Each process gets a range.
            results = pool.starmap(self._prime_number_avx2_range, [(duration, core_id) for core_id in range(num_cores)])
            #Combine the results from all the processes
            for result in results:
                ops_count += result[0]
                error_count += result[1]
                primes_found += result[2]
        return ops_count, error_count, primes_found

    def _prime_number_avx2_range(self, duration, core_id):
        """Calculates prime numbers within a range using AVX2 instructions"""
        start_time = time.time()
        end_time = start_time + duration
        ops_count = 0
        error_count = 0
        primes_found = 0 #Keep track of primes

        # Ensure the ranges don't overlap
        range_start = 2 + core_id * 10000  # Example: Each core gets a range of 10000
        range_end = range_start + 10000
        while time.time() < end_time:
            for num in range(range_start, range_end):

                # Create an AVX register with 8 copies of the number
                num_vec = _mm256_set1_epi32(num)
                is_prime_avx = True

                # Check divisibility up to sqrt(num)
                for i in range(2, int(num**0.5) + 1):
                    divisor_vec = _mm256_set1_epi32(i)
                    # Perform division
                    quotient_vec = _mm256_div_epi32(num_vec, divisor_vec)
                    # Multiply back to check for remainder
                    remainder_vec = _mm256_mullo_epi32(quotient_vec, divisor_vec)
                    # Check for zero remainder
                    comparison_result = _mm256_cmpeq_epi32(remainder_vec, _mm256_setzero_si256())
                    # If any remainder is zero, it's not prime
                    if _mm256_movemask_epi8(comparison_result) != 0:
                        is_prime_avx = False
                        break
                if is_prime_avx:
                    primes_found += 1
                ops_count += 1 #Count loop iterations as operations
        return ops_count, error_count, primes_found

    def prime_number_avx512(self, duration):
        """Calculates prime numbers with AVX512"""
        #Find out number of cores.
        num_cores = multiprocessing.cpu_count()
        #Variable to keep track of counts across threads.
        ops_count = 0
        error_count = 0
        primes_found = 0

        #Create a pool of worker processes.
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Divide work among processes. Each process gets a range.
            results = pool.starmap(self._prime_number_avx512_range, [(duration, core_id) for core_id in range(num_cores)])
            #Combine the results from all the processes
            for result in results:
                ops_count += result[0]
                error_count += result[1]
                primes_found += result[2]
        return ops_count, error_count, primes_found
    
    def _prime_number_avx512_range(self, duration, core_id):
        """Calculates prime numbers within a range using AVX512 instructions"""
        start_time = time.time()
        end_time = start_time + duration
        ops_count = 0
        error_count = 0
        primes_found = 0 #Keep track of primes

        # Ensure the ranges don't overlap
        range_start = 2 + core_id * 10000  # Example: Each core gets a range of 10000
        range_end = range_start + 10000
        while time.time() < end_time:
            for num in range(range_start, range_end):
                # Create an AVX-512 register with 16 copies of the number
                num_vec = _mm512_set1_epi32(num)
                is_prime_avx = True

                # Check divisibility up to sqrt(num)
                for i in range(2, int(num**0.5) + 1):
                    divisor_vec = _mm512_set1_epi32(i)
                    # Perform integer division
                    quotient_vec = _mm512_div_epi32(num_vec, divisor_vec)
                    # Multiply back to check for remainder
                    remainder_vec = _mm512_mullo_epi32(quotient_vec, divisor_vec)
                    # Compare with original number to check for divisibility
                    comparison_result = _mm512_cmpeq_epi32_mask(remainder_vec, num_vec)
                    # If any comparison is true (non-zero mask), the number is not prime
                    if comparison_result != 0:
                        is_prime_avx = False
                        break
                if is_prime_avx:
                    primes_found += 1
                ops_count += 1 #Count loop iterations as operations
        return ops_count, error_count, primes_found

    def run_floating_point_test(self):
        """Performs the Floating Point test."""
        try:
            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                details = "OS Not Supported"
                self.result_labels["Floating Point"].config(text=result, foreground=color)
                self.write_log_file("Floating Point", result, details)
                return

            # --- Get CPU Information ---
            cpu_info = cpuinfo.get_cpu_info()
            avx_flags = cpu_info.get('flags', [])

            # Determine Max AVX Support (same logic as Prime Number)
            avx_support_level = 0
            if 'avx512f' in avx_flags:
                avx_support_level = 3
            elif 'avx2' in avx_flags:
                avx_support_level = 2
            elif 'avx' in avx_flags:
                avx_support_level = 1

            # --- Run the Floating Point Test ---
            mflops, error_count = self.run_fp_calculation(self.fp_timer, avx_support_level)

            # --- Determine Result and Update Label ---
            if self.display_only_fp:
                result = "NO COMPARE"
                color = "blue"
                details = (f"MFLOPS: {mflops:.2f}, "
                           f"Errors: {error_count} (No Comparison)")
            elif error_count == 0:
                result = "PASS"
                color = "green"
                details = f"MFLOPS: {mflops:.2f}, Errors: {error_count}"
            else:
                result = "FAIL"
                color = "red"
                details = f"MFLOPS: {mflops:.2f}, Errors: {error_count}"

            self.result_labels["Floating Point"].config(text=result, foreground=color)
            self.write_log_file("Floating Point", result, details)


        except Exception as e:
            self.result_labels["Floating Point"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"Floating Point Test Error: {e}")
            self.write_log_file("Floating Point", "ERROR", str(e))


    def run_fp_calculation(self, duration, avx_level=0):
        """Runs the floating-point calculation test."""
        start_time = time.time()
        end_time = start_time + duration
        ops_count = 0
        error_count = 0
        n = 2500

        #Basic Floating Point operations
        while time.time() < end_time:
            if avx_level == 3:
                ops_count, error_count = self.fp_avx512(n, ops_count, error_count)
            elif avx_level == 2 or avx_level == 1:
                ops_count, error_count = self.fp_avx(n, ops_count, error_count)
            else:
                ops_count, error_count = self.fp_operations(n, ops_count, error_count)
            if self.stop_on_error_fp and error_count > 0:
                break #Stop if errors are found and the option is enabled.

        end_time = time.time()  # Get the *actual* end time
        elapsed_time = end_time - start_time
        #Prevent division by zero error
        mflops = (ops_count * 13) / 1_000_000 / elapsed_time if elapsed_time >0 else 0  # Calculate MFLOPS (Millions of Floating-Point Operations Per Second)

        return mflops, error_count

    def fp_operations(self, n, ops_count, error_count):
        """Performs basic floating point operations."""
        a = 22345678.1231234567890
        b = 12234678.1231234567890
        c = -12345.1231234567890
        for i in range(int(n)):  # Use range for looping
            a += i * 992200999001234.567890
            b += i * 992200999001234.567890
            c += i * 992200999001234.567890

            eq1 = a + b + c
            eq2 = c + b + a
            if not math.isclose(eq1, eq2, rel_tol=1e-9):
                error_count += 1

            eqmul1 = a * b * c
            eqmul2 = c * b * a
            if not math.isclose(eqmul1, eqmul2, rel_tol=1e-9):
                error_count += 1

            eq1mul1 = 1.0 * a * b
            eq1mul2 = a * 1.0 * b
            eq1mul3 = a * b * 1.0
            eq1mul4 = a * b
            if not (math.isclose(eq1mul1, eq1mul2, rel_tol=1e-9) and
                    math.isclose(eq1mul2, eq1mul3, rel_tol=1e-9) and
                    math.isclose(eq1mul3, eq1mul4, rel_tol=1e-9)):
                error_count += 1

            eqmuldiv1 = a / 1.0 * b
            eqmuldiv2 = a * b / 1.0
            eqmuldiv3 = a * b
            if not (math.isclose(eqmuldiv1, eqmuldiv2, rel_tol=1e-9) and
                    math.isclose(eqmuldiv2, eqmuldiv3, rel_tol=1e-9)):
                error_count += 1

            eqmuladd1 = 2.0 * a
            eqmuladd2 = a + a
            if not math.isclose(eqmuladd1, eqmuladd2, rel_tol=1e-9):
                error_count += 1

        ops_count += 1 #Counts the outer loop
        return ops_count, error_count

    def fp_avx(self, n, ops_count, error_count):
      """Performs floating points operations using AVX instructions"""
      a = 22345678.1231234567890
      b = 12234678.1231234567890
      c = -12345.1231234567890
      for i in range(int(n)):
          a += i * 992200999001234.567890
          b += i * 992200999001234.567890
          c += i * 992200999001234.567890

          a_vec = _mm256_set1_pd(a)
          b_vec = _mm256_set1_pd(b)
          c_vec = _mm256_set1_pd(c)

          add_vec1 = _mm256_add_pd(a_vec, b_vec)
          add_vec2 = _mm256_add_pd(add_vec1, c_vec)
          eq1 = _mm256_extract_f64(add_vec2, 0)  # Extract result

          add_vec_rev1 = _mm256_add_pd(c_vec, b_vec)
          add_vec_rev2 = _mm256_add_pd(add_vec_rev1, a_vec)
          eq2 = _mm256_extract_f64(add_vec_rev2, 0)  # Extract result

          if not math.isclose(eq1, eq2, rel_tol=1e-9):
              error_count += 1

          mul_vec1 = _mm256_mul_pd(a_vec, b_vec)
          mul_vec2 = _mm256_mul_pd(mul_vec1, c_vec)
          eqmul1 = _mm256_extract_f64(mul_vec2, 0)

          mul_vec_rev1 = _mm256_mul_pd(c_vec, b_vec)
          mul_vec_rev2 = _mm256_mul_pd(mul_vec_rev1, a_vec)
          eqmul2 = _mm256_extract_f64(mul_vec_rev2, 0)

          if not math.isclose(eqmul1, eqmul2, rel_tol=1e-9):
              error_count += 1
          one_vec = _mm256_set1_pd(1.0)
          mul1_vec1 = _mm256_mul_pd(one_vec, a_vec)
          mul1_vec2 = _mm256_mul_pd(mul1_vec1, b_vec)
          eq1mul1 = _mm256_extract_f64(mul1_vec2, 0)

          mul1_vec_rev1 = _mm256_mul_pd(a_vec, one_vec)
          mul1_vec_rev2 = _mm256_mul_pd(mul1_vec_rev1, b_vec)
          eq1mul2 = _mm256_extract_f64(mul1_vec_rev2, 0)


          mul1_vec3 = _mm256_mul_pd(a_vec, b_vec)
          mul1_vec4 = _mm256_mul_pd(mul1_vec3, one_vec)
          eq1mul3 = _mm256_extract_f64(mul1_vec4, 0)


          mul1_vec5 = _mm256_mul_pd(a_vec, b_vec)
          eq1mul4 = _mm256_extract_f64(mul1_vec5, 0)

          if not (math.isclose(eq1mul1, eq1mul2, rel_tol=1e-9) and
                  math.isclose(eq1mul2, eq1mul3, rel_tol=1e-9) and
                  math.isclose(eq1mul3, eq1mul4, rel_tol=1e-9)):
              error_count += 1
          muldiv_vec1 = _mm256_div_pd(a_vec, one_vec)
          muldiv_vec2 = _mm256_mul_pd(muldiv_vec1, b_vec)
          eqmuldiv1 = _mm256_extract_f64(muldiv_vec2, 0)
          muldiv_vec3 = _mm256_mul_pd(a_vec, b_vec)
          muldiv_vec4 = _mm256_div_pd(muldiv_vec3, one_vec)
          eqmuldiv2 = _mm256_extract_f64(muldiv_vec4, 0)

          muldiv_vec5 = _mm256_mul_pd(a_vec, b_vec)
          eqmuldiv3 = _mm256_extract_f64(muldiv_vec5, 0)


          if not (math.isclose(eqmuldiv1, eqmuldiv2, rel_tol=1e-9) and
                  math.isclose(eqmuldiv2, eqmuldiv3, rel_tol=1e-9)):
              error_count += 1

          two_vec = _mm256_set1_pd(2.0)
          muladd_vec1 = _mm256_mul_pd(two_vec, a_vec)
          eqmuladd1 = _mm256_extract_f64(muladd_vec1, 0)

          muladd_vec2 = _mm256_add_pd(a_vec, a_vec)
          eqmuladd2 = _mm256_extract_f64(muladd_vec2, 0)
          if not math.isclose(eqmuladd1, eqmuladd2, rel_tol=1e-9):
              error_count += 1
      ops_count += 1
      return ops_count, error_count

    def fp_avx512(self, n, ops_count, error_count):
      """Performs floating points operations using AVX512 instructions"""
      a = 22345678.1231234567890
      b = 12234678.1231234567890
      c = -12345.1231234567890
      for i in range(int(n)):
          a += i * 992200999001234.567890
          b += i * 992200999001234.567890
          c += i * 992200999001234.567890

          a_vec = _mm512_set1_pd(a)
          b_vec = _mm512_set1_pd(b)
          c_vec = _mm512_set1_pd(c)

          add_vec1 = _mm512_add_pd(a_vec, b_vec)
          add_vec2 = _mm512_add_pd(add_vec1, c_vec)
          eq1 = _mm512_extract_f64(add_vec2, 0)  # Extract result

          add_vec_rev1 = _mm512_add_pd(c_vec, b_vec)
          add_vec_rev2 = _mm512_add_pd(add_vec_rev1, a_vec)
          eq2 = _mm512_extract_f64(add_vec_rev2, 0)  # Extract result

          if not math.isclose(eq1, eq2, rel_tol=1e-9):
              error_count += 1

          mul_vec1 = _mm512_mul_pd(a_vec, b_vec)
          mul_vec2 = _mm512_mul_pd(mul_vec1, c_vec)
          eqmul1 = _mm512_extract_f64(mul_vec2, 0)

          mul_vec_rev1 = _mm512_mul_pd(c_vec, b_vec)
          mul_vec_rev2 = _mm512_mul_pd(mul_vec_rev1, a_vec)
          eqmul2 = _mm512_extract_f64(mul_vec_rev2, 0)

          if not math.isclose(eqmul1, eqmul2, rel_tol=1e-9):
              error_count += 1

          one_vec = _mm512_set1_pd(1.0)
          mul1_vec1 = _mm512_mul_pd(one_vec, a_vec)
          mul1_vec2 = _mm512_mul_pd(mul1_vec1, b_vec)
          eq1mul1 = _mm512_extract_f64(mul1_vec2, 0)

          mul1_vec_rev1 = _mm512_mul_pd(a_vec, one_vec)
          mul1_vec_rev2 = _mm512_mul_pd(mul1_vec_rev1, b_vec)
          eq1mul2 = _mm512_extract_f64(mul1_vec_rev2, 0)

          mul1_vec3 = _mm512_mul_pd(a_vec, b_vec)
          mul1_vec4 = _mm512_mul_pd(mul1_vec3, one_vec)
          eq1mul3 = _mm512_extract_f64(mul1_vec4, 0)

          mul1_vec5 = _mm512_mul_pd(a_vec, b_vec)
          eq1mul4 = _mm512_extract_f64(mul1_vec5, 0)

          if not (math.isclose(eq1mul1, eq1mul2, rel_tol=1e-9) and
                  math.isclose(eq1mul2, eq1mul3, rel_tol=1e-9) and
                  math.isclose(eq1mul3, eq1mul4, rel_tol=1e-9)):
              error_count += 1

          muldiv_vec1 = _mm512_div_pd(a_vec, one_vec)
          muldiv_vec2 = _mm512_mul_pd(muldiv_vec1, b_vec)
          eqmuldiv1 = _mm512_extract_f64(muldiv_vec2, 0)

          muldiv_vec3 = _mm512_mul_pd(a_vec, b_vec)
          muldiv_vec4 = _mm512_div_pd(muldiv_vec3, one_vec)
          eqmuldiv2 = _mm512_extract_f64(muldiv_vec4, 0)

          muldiv_vec5 = _mm512_mul_pd(a_vec, b_vec)
          eqmuldiv3 = _mm512_extract_f64(muldiv_vec5, 0)

          if not (math.isclose(eqmuldiv1, eqmuldiv2, rel_tol=1e-9) and
                  math.isclose(eqmuldiv2, eqmuldiv3, rel_tol=1e-9)):
              error_count += 1

          two_vec = _mm512_set1_pd(2.0)
          muladd_vec1 = _mm512_mul_pd(two_vec, a_vec)
          eqmuladd1 = _mm512_extract_f64(muladd_vec1, 0)

          muladd_vec2 = _mm512_add_pd(a_vec, a_vec)
          eqmuladd2 = _mm512_extract_f64(muladd_vec2, 0)

          if not math.isclose(eqmuladd1, eqmuladd2, rel_tol=1e-9):
              error_count += 1
      ops_count += 1
      return ops_count, error_count

    def run_math_fma_test(self):
        """Performs the FMA3 (Fused Multiply-Add) test."""
        try:
            # Check for supported OS (Linux)
            if platform.system() != "Linux":
                result = "FAIL"
                color = "red"
                details = "OS Not Supported"
                self.result_labels["Math"].config(text=result, foreground=color)
                self.write_log_file("Math", result, details)
                return

            # --- Check for FMA3 Support (using cpuinfo) ---
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])

            fma3_supported = 'fma3' in flags
            avx_supported = any(flag in flags for flag in ['avx', 'avx2', 'avx512f']) # Check for any AVX support

            if not avx_supported:
                result = "NOT SUPPORTED"
                color = "orange"  # Use orange for warnings/not supported
                details = "AVX is required for FMA3 testing, but is not supported by the OS or CPU."
                self.result_labels["Math"].config(text=result, foreground=color)
                self.write_log_file("Math", result, details)
                return

            if not fma3_supported:
                result = "NOT SUPPORTED"
                color = "orange"
                details = "FMA3 is not supported by this CPU."
                self.result_labels["Math"].config(text=result, foreground=color)
                self.write_log_file("Math", result, details)
                return
            
            # --- Run the FMA3 Test ---
            test_passed = self.run_fma_test(self.fma_timer)  # Call the test function

            # --- Determine Result and Update Label ---
            if test_passed:
                result = "PASS"
                color = "green"
                details = "FMA3 test passed."
            else:
                result = "FAIL"
                color = "red"
                details = "FMA3 test failed."

            self.result_labels["Math"].config(text=result, foreground=color)
            self.write_log_file("Math", result, details)

        except Exception as e:
            self.result_labels["Math"].config(text="ERROR", foreground="red")
            messagebox.showerror("Error", f"FMA3 Test Error: {e}")
            self.write_log_file("Math", "ERROR", str(e))

    def run_fma_test(self, duration):
        """Executes the FMA3 test for the specified duration."""
        start_time = time.time()
        end_time = start_time + duration
        test_passed = True

        loopsize = 19000000
        loopstart = 18990000
        
        # Use numpy arrays for efficient calculations.
        a = np.zeros(loopsize, dtype=np.float32)
        b = np.zeros(loopsize, dtype=np.float32)
        c = np.zeros(loopsize, dtype=np.float32)

        while time.time() < end_time:
            # Initialize arrays with values based on the loop index 'i'
            for i in range(loopstart, loopsize):
                # Use smaller values to prevent overflow/underflow
                a[i] = i * 0.000001 
                b[i] = i * 0.000001
                c[i] = i * 0.000001
            #Perform FMA and C calculations using vectorized operations with numpy.
            calc_fma = np.fma(a[loopstart:loopsize], b[loopstart:loopsize], c[loopstart:loopsize])
            calc_c = a[loopstart:loopsize] * b[loopstart:loopsize] + c[loopstart:loopsize]
            
            # Compare results element-wise
            if not np.allclose(calc_fma, calc_c, rtol=1e-5, atol=1e-8): # Use numpy's allclose
                test_passed = False
                break # Exit the loop on failure

        return test_passed

    def write_log_file(self, module_name, result, details):
        """Writes test results to a log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("processor_test_log.txt", "a") as log_file:
            log_file.write(f"[{timestamp}] {module_name}: {result} - {details}\n")


    def run_all_tests(self):
        for module in self.test_modules:
            self.run_test(module)

    def update_overall_status(self):
        all_passed = True
        for module in self.test_modules:
            if self.result_labels[module]["text"] == "FAIL":
                all_passed = False
                break
            elif self.result_labels[module]["text"] == "Not Tested":
                self.status_label.config(text="Overall Status: Pending", foreground="black")
                return
            elif self.result_labels[module]["text"] == "ERROR":
                all_passed = False
                break
            # Consider WARNING as a failure for overall status
            elif self.result_labels[module]["text"] == "WARNING":
                all_passed = False
                break
            elif self.result_labels[module]["text"] == "NOT SUPPORTED":
                all_passed = False
                break

        if all_passed:
            self.status_label.config(text="Overall Status: PASS", foreground="green")
        else:
            self.status_label.config(text="Overall Status: FAIL", foreground="red")

# Important Main code
if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessorTesterApp(root)
    root.mainloop()
