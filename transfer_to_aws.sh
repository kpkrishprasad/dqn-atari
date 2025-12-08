#!/bin/bash

# Transfer DQN project to AWS EC2 instance

KEY_FILE=~/.ssh/your-key.pem
EC2_USER=ec2-user
EC2_IP=YOUR_EC2_IP
REMOTE_DIR=/home/ec2-user/dqn-atari

echo "Creating remote directory..."
ssh -i $KEY_FILE $EC2_USER@$EC2_IP "mkdir -p $REMOTE_DIR"

echo "Transferring files to AWS..."
rsync -avz --progress \
  -e "ssh -i $KEY_FILE" \
  --exclude 'logs/' \
  --exclude '*.pyc' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'atari_model.pack' \
  --exclude '.DS_Store' \
  ./ $EC2_USER@$EC2_IP:$REMOTE_DIR/

echo "Transfer complete!"
echo "To connect: ssh -i $KEY_FILE $EC2_USER@$EC2_IP"
