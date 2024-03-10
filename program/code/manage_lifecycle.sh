# Define your variables
$DOMAIN_ID = "<your-domain-id>"
$USER_PROFILE = "<your-user-profile>"
$CODE_FOLDER = "<path-to-your-packages.sh>"

# Encode the content of packages.sh to Base64
$LCC_CONTENT = Get-Content "$CODE_FOLDER\packages.sh" | Out-String | openssl base64 -A

# Delete the existing SageMaker Studio Lifecycle Config if it exists
aws sagemaker delete-studio-lifecycle-config --studio-lifecycle-config-name packages

# Create a new SageMaker Studio Lifecycle Config
$response = aws sagemaker create-studio-lifecycle-config `
    --studio-lifecycle-config-name packages `
    --studio-lifecycle-config-content $LCC_CONTENT `
    --studio-lifecycle-config-app-type KernelGateway | ConvertFrom-Json

$arn = $response.StudioLifecycleConfigArn
Write-Host "ARN: $arn"

# Update the SageMaker user profile with the new Lifecycle Config ARN
aws sagemaker update-user-profile `
    --domain-id $DOMAIN_ID `
    --user-profile-name $USER_PROFILE `
    --user-settings @{
        KernelGatewayAppSettings = @{
            LifecycleConfigArns = @($arn)
        }
    } | ConvertTo-Json