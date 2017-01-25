---
published: false
---
## Detailed Guide to AWS CodeDeploy Installation

Create Bucket with policy from bucket-policy.txt, be sure to change the principal for your account

Create the service role for CodeDeploy to allow it read the EC2 tags
Service Role/create-role.bat

Attach the managed policy AWSCodeDeployRole to allow the reading of EC2 tags
Service Role/attach-role-policy.bat

Create the instance profile for the EC2 instance to be able to pull objects from S3
InstanceProfile/create-role.bat
InstanceProfile/put-role-policy.bat
InstanceProfile/create-instance-profile.bat

Create the EC2 instance with the Instance Profile associated, assign this EC2 instance the tags Key=Name,Value=CodeDeployDemo

Install the CodeDeploy Agent on the EC2 instance
Set-ExecutionPolicy RemoteSigned
Import-Module AWSPowerShell
New-Item –Path "c:\temp" –ItemType "directory" -Force
powershell.exe -Command Read-S3Object -BucketName aws-codedeploy-us-west-1  -Key latest/codedeploy-agent.msi -File c:\temp\codedeploy-agent.msi
c:\temp\codedeploy-agent.msi /quiet /l c:\temp\host-agent-install-log.txt

Create the CodeDeploy Application
create-application.bat

Create the deployment group specifying the ARN of the ServiceRole, be sure to change the ARN on the script
create-deployment-group.bat

When a revision is ready run this script to zip the folder and push it to the BT Bucket
BTApplication/deploy-push.bat

To start the deployment of the uploaded object on S3 run:
create-deployment.bat

