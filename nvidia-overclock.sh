#!/bin/bash
# Enable persistence mode
nvidia-smi -pm 1

# Set PowerMizer mode to "Prefer Maximum Performance"
nvidia-settings -a '[gpu:0]/GpuPowerMizerMode=1'

# Set GPU fan control to manual and set fan speed to 70%
nvidia-settings -a '[gpu:0]/GPUFanControlState=0'

# Set GPU core clock offset to +100 MHz
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[3]=60'

# Set GPU memory clock offset to +500 MHz
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[3]=1000'

