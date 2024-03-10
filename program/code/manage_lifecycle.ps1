# Define your variables
$DOMAIN_ID = "<your-domain-id>"
$USER_PROFILE = "<your-user-profile>"
$CODE_FOLDER = "<path-to-your-packages.sh>"
$REGION = "eu-north-1"

# Ensure the full path to packages.sh is correct
$PACKAGE_FILE_PATH = Join-Path -Path $CODE_FOLDER -ChildPath "packages.sh"

# Encode the content of packages.sh to Base64
$LCC_CONTENT = Get-Content $PACKAGE_FILE_PATH -Raw | openssl base64 -A

# Try to delete the existing SageMaker Studio Lifecycle Config if it exists
try {
    aws sagemaker delete-studio-lifecycle-config --studio-lifecycle-config-name packages --region $REGION --profile $USER_PROFILE
} catch {
    Write-Host "No existing lifecycle config to delete or error deleting. Continuing..."
}

# Create a new SageMaker Studio Lifecycle Config
$response = aws sagemaker create-studio-lifecycle-config `
    --studio-lifecycle-config-name packages `
    --studio-lifecycle-config-content $LCC_CONTENT `
    --studio-lifecycle-config-app-type KernelGateway `
    --region $REGION `
    --profile $USER_PROFILE | ConvertFrom-Json

$arn = $response.StudioLifecycleConfigArn
Write-Host "ARN: $arn"

# Construct the user settings JSON string correctly
$userSettingsJson = @"
{
    \"KernelGatewayAppSettings\": {
        \"LifecycleConfigArns\": [\"$arn\"]
    }
}
"@

# Update the SageMaker user profile with the new Lifecycle Config ARN
aws sagemaker update-user-profile --domain-id $DOMAIN_ID `
    --user-profile-name $USER_PROFILE `
    --user-settings $userSettingsJson `
    --region $REGION `
    --profile $USER_PROFILE