# Usa a AWAS base image for Python 2.12 
from public.ecr.aws/lambda/python:3.12 

#Install build-essential complier and tools 
RUN microdnf update -y && microdnf install -y gcc-c++ make 

#Copy requirements.txt 
COPY requirements.txt ${LAMBDA_TASK_ROOT}

#Install packages 
RUN pip install -r requirements.txt 

#Copy functions code 
COPY travel-agent.py ${LAMBDA_TASK_ROOT}

#Set the permissions to make the file executable
RUN chmod +x travel-agent.py

#Set the CMD to your handler
CMD ["travel-agent.lambda_handler"]




