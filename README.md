## AWS pipline
This project implements an end-to-end machine learning pipeline for predicting football match outcomes using AWS cloud infrastructure. The pipeline covers data ingestion, preprocessing, model training, evaluation, and deployment, utilizing modern tools and services to ensure scalability and efficiency. The core model is built using XGBoost, and the entire pipeline is orchestrated using various AWS services for automation and reliability.

**Key Technologies**:
- Python 3.12: Core programming language for the project.
- XGBoost: Machine learning algorithm for training the prediction model.
- AWS S3: For storing datasets and model artifacts.
- AWS SageMaker: For training, tuning, and deploying machine learning models.
- AWS Lambda: For scalable, serverless execution of tasks.
- AWS Step Functions: To orchestrate the end-to-end pipeline.
- AWS CloudWatch: For monitoring and logging the pipeline processes.

This project is designed to automate the entire machine learning workflow from data preparation to model deployment, ensuring scalability and efficiency in production environments.


## Installation

```
git clone https://github.com/kamil0920/aws_football_predictor.git
```


In main directory create .env:

### `.env` File Instructions:

1. **BUCKET**:
   - Set the name for your S3 bucket, which will be used for storing data. You can create a new bucket or use an existing one.
   - Example: `my-ml-bucket`.

2. **DOMAIN_ID**:
   - In the AWS Console, navigate to **SageMaker**. Under **Control Panel** or **Domains**, you will find your **Domain ID**.
   - Example: `d-xxxxxxxxx`.

3. **ACCOUNT_ID**:
   - Your **Account ID** is a 12-digit number. You can find it at the top right of the AWS Console under **My Account** or use the AWS CLI to retrieve it.
   - Example: `123456789012`.

4. **USER_NAME**:
   - Specify the **developer user** name, which will be used for most of the operations.
   - Example: `developer-user`.

5. **AWS_REGION**:
   - You can find your AWS region in the AWS Console (top right corner). This is where your SageMaker resources will be hosted.
   - Example: `us-west-2`.

6. **ROLE_NAME**:
   - Set the name for your SageMaker execution role, which will be used for running your ML jobs in SageMaker. This should be descriptive for your project.
   - Example: `FootballProjectExecutionRole`.

7. **POLICY_NAME**:
   - Set the name for your custom policy, which will define the permissions for the execution role.
   - Example: `FootballProjectExecutionPolicy`.

8. **MODEL_PACKAGE_GROUP**:
   - This is the name you use to group different versions of your SageMaker model. Use a meaningful name that describes your project or model group.
   - Example: `FootballModelGroup`.

9. **DATA_FILEPATH_X**:
   - Provide the **local path** to your input dataset file (features file), such as the CSV file containing your training data.
   - Example: `/path/to/data.csv`.

10. **DATA_FILEPATH_Y**:
    - Provide the **local path** to your target variable dataset file (labels file), such as the CSV file containing the labels for your training data.
    - Example: `/path/to/target.csv`.

11. **ACCESS_KEY**:
    - Provide your **AWS access key** which is used to authenticate and interact with AWS services, including SageMaker.
    - Example: `AKIAIOSFODNN7EXAMPLE`.

12. **SECRET_KEY**:
    - Provide your **AWS secret access key** which, together with the access key, is used to authenticate your requests to AWS services.
    - Example: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`.

---

### Final `.env` Example:

```plaintext
BUCKET=my-ml-bucket
DOMAIN_ID=d-xxxxxxxxx
ACCOUNT_ID=123456789012
USER_NAME=developer-user
AWS_REGION=us-west-2
ROLE_NAME=FootballProjectExecutionRole
POLICY_NAME=FootballProjectExecutionPolicy
MODEL_PACKAGE_GROUP=FootballModelGroup

DATA_FILEPATH_X=/path/to/data.csv
DATA_FILEPATH_Y=/path/to/target.csv
```

To set up the project, execute this notebook: [initial_setup.ipynb](program/code/config/initial_setup.ipynb).
After completing the setup, you can run the main end-to-end pipeline [main_pipeline.ipynb](program%2Fmain_pipeline.ipynb).