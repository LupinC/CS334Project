# CS334Project

## purpose
The purpose is to predict whether someone will have a heart attack or not and the dataset is a kaggle dataset https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease.

## Deployment on aws ec2
1. install docker `sudo yum install docker` and start docker `sudo systemctl start docker`
2. install nginx `sudo yum install nginx`
3. change nginx configration `sudo nano /etc/nginx/nginx.conf`for server to:

   server {
     listen 80;
     listen [::]:80;
     server_name _;

     location / {
         proxy_pass http://localhost:8501;
         proxy_http_version 1.1;
         proxy_set_header Upgrade $http_upgrade;
         proxy_set_header Connection "upgrade";
         proxy_set_header Host $host;
     }
   }
  
  
4. reconfig nginx `sudo systemctl restart nginx`
5. git clone https://github.com/LupinC/CS334Project.git
6. modify the code inside if you want, follow the comment if needed
7. `cd projectdirectory` and build docker image/container: `sudo docker build -t giveitanameyoulike .`
8. load up the website `sudo docker run -p 8501:8501 giveitanameyoulike`
