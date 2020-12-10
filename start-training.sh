#!/bin/bash 

NR_INSTANCES=4
declare -a INSTANCES

for SUBNET_NR in $(seq 1 $NR_INSTANCES);
do 
  sed -i "51s/.*/NR_INSTANCES=${NR_INSTANCES}/" ec2-user-data-bash.sh
  sed -i "52s/.*/SUBNET_NR=${SUBNET_NR}/" ec2-user-data-bash.sh

  INSTANCES[$SUBNET_NR]=$(aws ec2 run-instances \
                            --image-id ami-0f3fd8037b0b9df52 \
                            --iam-instance-profile Name="bnn-train-profile" \
                            --subnet-id subnet-22098d44 \
                            --count 1 \
                            --instance-type c4.large \
                            --key-name admin-orion \
                            --security-group-ids sg-05ee90dce2dfbec77 \
                            --user-data file://ec2-user-data-bash.sh | jq -r '.Instances[].InstanceId') 
done 

echo "Monitoring instances" ${INSTANCES[@]}
# while true;
# do
#   # Monitor instance statuses at every 2 minutes 
#   TERMINATED_INSTANCES=(aws ec2 describe-instances-status \
#                           --instance-ids ${INSTANCES[@]} \
#                           --filters "Name=instance-state-name,Values=shutting-down,terminated" | jq -r '.InstanceStatuses[].InstanceId')

#   # Check that instances haven't shut down prematurely 


#   sleep 120;
# done 
