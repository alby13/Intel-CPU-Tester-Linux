<img src="https://raw.githubusercontent.com/alby13/Intel-CPU-Tester-Linux/refs/heads/main/screenshot.jpg">

# Intel CPU Tester for Linux
Linux Intel Processor Tester is meant to be a Linux option to the Windows version of Intel Processor Diagnostic Tool (IPDT)

Written in Python 3.10+
<br><br>

## Please note: 

This is an early release and untested version of the program, meant to serve as a missing tool for Linux.
<br><br>

Requirements:
```
py-cpuinfo
psutil
```
<br>

After running, a log file will be created: processor_test_log.txt
<br><br>

## Things that are checked:

- Processor Name
- Architecture
- Number of processors (logical)
- Number of physical cores
- Vendor ID
- L1, L2, & L3 Cache

## Things that are tested:

- Genuine Intel Processor check looking at Vendor ID reporting
- Brand String -  Testing results of BrandString query
- Cache performance and stability test
- IMMXSSE (Work In Progress/Simulated in Program)
- IMC (Integrated Memory Controller)
- Prime Number Math Check
- Floating Point Math Check
-  Math Test - FMA3 (Fused Multiply-Add 3-operand)<br>

# Intel CPU Cache Memory Latency Checker

Available for both Linux and WIndows (MLC.exe):

https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html
<br><br>
 
## Software License:

GNU General Public License (GPL) 3.0
