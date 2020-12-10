# Update AWS CLI 
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
SYMLINK=$(which aws)
SYMLINK_POINT=$(ls -l /usr/local/bin/aws)
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update