#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# AWS Deployment Script — Agentic AI ROI Dashboard
# Deploys to AWS App Runner with Bedrock access via IAM role.
#
# Prerequisites:
#   1. AWS CLI v2 installed and configured (aws configure / SSO login)
#   2. Docker installed and running
#   3. Sufficient IAM permissions (ECR, App Runner, IAM)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ============================================================================

APP_NAME="roi-dashboard"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"
IAM_ROLE_NAME="${APP_NAME}-apprunner-role"
BEDROCK_POLICY_NAME="${APP_NAME}-bedrock-access"

echo "================================================"
echo "  Deploying ${APP_NAME} to AWS App Runner"
echo "  Account:  ${ACCOUNT_ID}"
echo "  Region:   ${AWS_REGION}"
echo "================================================"

# ── Step 1: Create ECR Repository ────────────────────────────────────────

echo ""
echo "[1/5] Creating ECR repository..."
aws ecr describe-repositories --repository-names "${APP_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${APP_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true

# ── Step 2: Build & Push Docker Image ────────────────────────────────────

echo ""
echo "[2/5] Building and pushing Docker image..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -t "${APP_NAME}:latest" .
docker tag "${APP_NAME}:latest" "${ECR_REPO}:latest"
docker push "${ECR_REPO}:latest"

# ── Step 3: Create IAM Role for App Runner with Bedrock Access ───────────

echo ""
echo "[3/5] Setting up IAM role with Bedrock permissions..."

TRUST_POLICY=$(cat <<'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "tasks.apprunner.amazonaws.com",
          "build.apprunner.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
TRUST
)

BEDROCK_POLICY=$(cat <<'BEDROCK'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.*"
    }
  ]
}
BEDROCK
)

ECR_ACCESS_POLICY=$(cat <<ECRPOL
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    }
  ]
}
ECRPOL
)

# Create role (skip if exists)
aws iam get-role --role-name "${IAM_ROLE_NAME}" 2>/dev/null || \
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document "${TRUST_POLICY}"

# Attach Bedrock policy
aws iam put-role-policy \
    --role-name "${IAM_ROLE_NAME}" \
    --policy-name "${BEDROCK_POLICY_NAME}" \
    --policy-document "${BEDROCK_POLICY}"

# Attach ECR access
aws iam put-role-policy \
    --role-name "${IAM_ROLE_NAME}" \
    --policy-name "${APP_NAME}-ecr-access" \
    --policy-document "${ECR_ACCESS_POLICY}"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE_NAME}"

echo "  IAM Role ARN: ${ROLE_ARN}"

# ── Step 4: Create/Update App Runner Service ─────────────────────────────

echo ""
echo "[4/5] Deploying to App Runner..."

SERVICE_EXISTS=$(aws apprunner list-services --region "${AWS_REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn" \
    --output text 2>/dev/null || echo "")

if [ -z "${SERVICE_EXISTS}" ]; then
    echo "  Creating new App Runner service..."
    aws apprunner create-service \
        --service-name "${APP_NAME}" \
        --region "${AWS_REGION}" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_REPO}:latest\",
                \"ImageRepositoryType\": \"ECR\",
                \"ImageConfiguration\": {
                    \"Port\": \"8501\"
                }
            },
            \"AutoDeploymentsEnabled\": true,
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ROLE_ARN}\"
            }
        }" \
        --instance-configuration "{
            \"Cpu\": \"1 vCPU\",
            \"Memory\": \"2 GB\",
            \"InstanceRoleArn\": \"${ROLE_ARN}\"
        }" \
        --health-check-configuration "{
            \"Protocol\": \"HTTP\",
            \"Path\": \"/_stcore/health\",
            \"Interval\": 10,
            \"Timeout\": 5,
            \"HealthyThreshold\": 1,
            \"UnhealthyThreshold\": 5
        }"
else
    echo "  Updating existing App Runner service..."
    aws apprunner start-deployment \
        --service-arn "${SERVICE_EXISTS}" \
        --region "${AWS_REGION}"
fi

# ── Step 5: Wait and display URL ─────────────────────────────────────────

echo ""
echo "[5/5] Waiting for deployment (this may take 3-5 minutes)..."

sleep 10

SERVICE_URL=$(aws apprunner list-services --region "${AWS_REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceUrl" \
    --output text 2>/dev/null || echo "pending...")

echo ""
echo "================================================"
echo "  Deployment initiated!"
echo ""
echo "  App URL: https://${SERVICE_URL}"
echo ""
echo "  Monitor status:"
echo "    aws apprunner list-services --region ${AWS_REGION}"
echo ""
echo "  View logs:"
echo "    aws apprunner list-operations \\"
echo "      --service-arn \$(aws apprunner list-services --region ${AWS_REGION} \\"
echo "        --query \"ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn\" \\"
echo "        --output text)"
echo ""
echo "  NOTE: Enable Bedrock model access in the AWS Console:"
echo "    1. Go to Amazon Bedrock > Model access"
echo "    2. Request access to Anthropic Claude 3.5 Sonnet"
echo "    3. Wait for approval (usually instant)"
echo "================================================"
