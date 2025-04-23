 serverless-geospatial
markdown
Copy
Edit
# Serverless Geospatial

This repository contains three serverless AWS Lambda services related to geospatial processing. Each folder is containerized using Docker and can be deployed using AWS ECR + Lambda.

## 📂 Project Structure

serverless-geospatial/ ├── dbscan/ │ ├── Dockerfile │ └── requirements.txt ├── lambda/ │ ├── Dockerfile │ └── requirements.txt └── lambda-kde-update/ ├── Dockerfile └── requirements.txt

yaml
Copy
Edit

Each folder is a standalone service with its own dependencies and Docker image.

---

## 🐳 Building Docker Images

Navigate into any folder (`dbscan`, `lambda`, or `lambda-kde-update`) and run the following:

```bash
# Build the Docker image
docker build -t <image-name> .
Replace <image-name> with a relevant name, e.g., lambda-dbscan.

🚀 Pushing to AWS ECR
1. Authenticate Docker to ECR
bash
Copy
Edit
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com
Replace:

<your-region> = e.g. us-east-1

<account-id> = your AWS account ID

2. Create an ECR repository (if not already created)
bash
Copy
Edit
aws ecr create-repository --repository-name <image-name> --region <your-region>
3. Tag and Push the Image
bash
Copy
Edit
# Tag the image
docker tag <image-name> <account-id>.dkr.ecr.<your-region>.amazonaws.com/<image-name>

# Push to ECR
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/<image-name>
🧠 Notes
Each Lambda function is packaged as a Docker container for flexible deployment.

Ensure your AWS CLI is configured with the correct IAM permissions.

Use AWS Lambda container image support to deploy.

