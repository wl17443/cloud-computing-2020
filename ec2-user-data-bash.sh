#!/bin/bash 

# Get Instance details 
# Get instance ID 
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=us-east-1

# Get Volume Id and availability zone
VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=BNN-datasets-results" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=BNN-datasets-results" --query "Volumes[].AvailabilityZone" --output text)

NEW_VOLUME_ID=$(aws ec2 create-volume \
				          --region $AWS_REGION \
				          --availability-zone $INSTANCE_AZ \
				          --snapshot-id snap-0a4fca20ac0b1ea01 \
				          --volume-type gp2 \
				          --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=BNN-datasets-checkpoints-temp}]' | jq -r '.VolumeId' )

aws ec2 wait volume-available --region $AWS_REGION --volume-id $NEW_VOLUME_ID

# Attach and mount snapshot 
aws ec2 attach-volume \
  --region $AWS_REGION \
  --volume-id $NEW_VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf
sleep 10 

# Mount volume and change ownership 
mkdir /training 
mount /dev/xvdf /training
chown -R root: /training/

echo "Getting training code..."
git clone https://github.com/wl17443/cloud-computing-2020.git
chown -R root: cloud-computing-2020
cd cloud-computing-2020

echo "Starting training..."
BATCH_SIZE=($(seq 5 5 20))
NR_BATCHES=($(seq 500 500 10000))
LEARNING_RATE=($(seq 5 5 100))
HIDDEN_LAYERS=($(seq 10 10 80))

NR_BATCH_SIZE=${#BATCH_SIZE[@]}
NR_NR_BATCHES=${#NR_BATCHES[@]}
NR_LEARNING_RATE=${#LEARNING_RATE[@]}
NR_HIDDEN_LAYERS=${#HIDDEN_LAYERS[@]}

NR_INSTANCES=4
SUBNET_NR=4

touch /training/Logs/log_file.txt

for ((i=$NR_BATCH_SIZE/$NR_INSTANCES*($SUBNET_NR-1); i<$NR_BATCH_SIZE/$NR_INSTANCES*$SUBNET_NR; i++));
do 
  for ((j=$NR_NR_BATCHES/$NR_INSTANCES*($SUBNET_NR-1); j<$NR_NR_BATCHES/$NR_INSTANCES*$SUBNET_NR; j++));
  do 
    for ((k=$NR_LEARNING_RATE/$NR_INSTANCES*($SUBNET_NR-1); k<$NR_LEARNING_RATE/$NR_INSTANCES*$SUBNET_NR; k++));
    do 
      for ((l=$NR_HIDDEN_LAYERS/$NR_INSTANCES*($SUBNET_NR-1); l<$NR_HIDDEN_LAYERS/$NR_INSTANCES*$SUBNET_NR; l++));
      do
        LEARNING_RATE_FLOAT=$(awk "BEGIN {print ${LEARNING_RATE[k]}/1000}")
        # echo "Training for parameters: ${BATCH_SIZE[i]}, ${NR_BATCHES[j]}, $LEARNING_RATE_FLOAT, and ${HIDDEN_LAYERS[l]}"
        sudo -H -u root bash -c "python backpropNN.py --batch_size ${BATCH_SIZE[i]} --nr_batches ${NR_BATCHES[j]} --learning_rate $LEARNING_RATE_FLOAT --hidden_layers ${HIDDEN_LAYERS[l]}"
      done 
    done 
  done 
done 

echo "Training complete..."
echo "Copying data from current EBS volume to central EBS Volume"

while true; do 
aws ec2 attach-volume \
  --region $AWS_REGION \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdg && break || aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID
done 
sleep 10 

# Mount volume and change ownership 
mkdir /databank 
xfs_admin -U generate /dev/xvdg
mount /dev/xvdg /databank
chown -R root: /databank/

echo "Copying result and log files to main volume..."
cp -a /training/Results/. /databank/Results
wait $!
cp -a /training/Logs/log_file.txt /databank/Logs/log_file$SUBNET_NR.txt
wait $!

echo "Unmounting and detaching main volume..."
umount -d /dev/xvdg
aws ec2 detach-volume --volume-id $VOLUME_ID --region $AWS_REGION
aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID

# Create snapshot of current EBS volume for backup 
aws ec2 create-snapshot \
  --volume-id $NEW_VOLUME_ID \
  --region $AWS_REGION \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=purpose,Value=training-data-backup}]'

umount -d /dev/xvdf
aws ec2 detach-volume --volume-id $NEW_VOLUME_ID --region $AWS_REGION
aws ec2 wait volume-available --region $AWS_REGION --volume-id $NEW_VOLUME_ID
# aws ec2 delete-volume --volume-id $NEW_VOLUME_ID --region $AWS_REGION

echo "Terminating instance..."

# Clean up after training
# Terminate ec2 instance after training
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID

# SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='ec2spot:fleet-request-id'].Value[]" --output text)
# aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances 
