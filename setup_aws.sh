#!/bin/bash

# Setup DQN environment on AWS EC2 instance
# Run this script ON THE AWS INSTANCE after transferring files

set -e  # Exit on error

echo "=== Setting up DQN environment on AWS ==="

# Update system packages
echo "Updating system packages..."
sudo yum update -y

# Install system dependencies
echo "Installing system dependencies..."
sudo yum install -y python3-pip python3-devel gcc gcc-c++ make git

# Install Python venv
echo "Creating Python virtual environment..."
cd ~/dqn-atari
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# Install Atari ROMs
echo "Installing Atari ROMs..."
pip install autorom[accept-rom-license]
AutoROM --accept-license

# Copy ROMs to ale_py directory
echo "Copying ROMs to ale_py..."
ROM_SOURCE=$(python3 -c "import site; print(site.getsitepackages()[0])")/AutoROM/roms
ROM_DEST=$(python3 -c "import site; print(site.getsitepackages()[0])")/ale_py/roms
mkdir -p $ROM_DEST
cp $ROM_SOURCE/*.bin $ROM_DEST/

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment in future sessions:"
echo "  cd ~/dqn-atari"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python dqn.py"
echo ""
echo "To monitor with tensorboard:"
echo "  tensorboard --logdir=./logs/atari_vanilla --host=0.0.0.0"
echo ""
