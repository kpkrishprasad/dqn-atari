#!/bin/bash

# Download trained model from AWS EC2 instance

KEY_FILE=~/.ssh/ask-hammerspace.pem
EC2_USER=ec2-user
EC2_IP=18.118.222.12
REMOTE_DIR=/home/ec2-user/dqn-atari
LOCAL_DIR=.

echo "=== Downloading trained model from AWS ==="

# Download the trained model
echo "Downloading atari_model.pack..."
scp -i $KEY_FILE $EC2_USER@$EC2_IP:$REMOTE_DIR/atari_model.pack $LOCAL_DIR/

# Optionally download logs for TensorBoard
read -p "Download logs folder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading logs..."
    rsync -avz --progress \
      -e "ssh -i $KEY_FILE" \
      $EC2_USER@$EC2_IP:$REMOTE_DIR/logs/ $LOCAL_DIR/logs/
fi

echo ""
echo "=== Download Complete! ==="
echo ""
echo "Model saved to: ./atari_model.pack"
echo ""
echo "To watch the agent play:"
echo "  python3 observe.py"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir=./logs/atari_vanilla"
echo ""
