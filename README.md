# AI Model for Multi-Class Text Classification

## Project Overview

This project is an AI model built on top of a DistilBERT Hugging Face model that classifies a given text title into one of four categories: **Science**, **Business**, **Health**, and **Entertainment**. The goal was to develop a multi-class text classification system that accurately predicts the category of a given title.

### Key Features

- **Multi-Class Classification**: The model can classify titles into one of four distinct categories.
- **Model Exploration**: Various Hugging Face models were explored to identify the most suitable pre-trained transformer for the task.
- **Fine-Tuning**: The chosen DistilBERT model was fine-tuned using a custom dataset tailored to the classification categories.
- **Deployment**: The model was deployed using AWS Lambda and AWS API Gateway, allowing for serverless and scalable API access.
- **Load Testing**: Load testing was conducted with thousands of requests to identify the most appropriate server configuration for optimal performance.

## Model Development

1. **Model Selection**: Different Hugging Face models were explored, and DistilBERT was selected based on its efficiency and performance for text classification tasks.
2. **Fine-Tuning**: The DistilBERT model was fine-tuned using a dataset of titles categorized into Science, Business, Health, and Entertainment. This process involved supervised learning to improve the model's accuracy in categorizing unseen titles.
3. **Training**: The model was trained with the goal of achieving high accuracy while avoiding overfitting, employing techniques such as learning rate optimization and batch size tuning.

## Deployment

The model was deployed using a serverless architecture:

- **AWS Lambda**: Provides scalability and reduced operational costs by running the model inference in a serverless environment.
- **AWS API Gateway**: Acts as the entry point for accessing the model through a REST API.

To determine the best server configuration, a load test was conducted by sending thousands of requests to evaluate performance, ensuring the deployment could handle high traffic and maintain low latency.

## How to Use

You can use the deployed API to classify titles by sending a POST request with the title as input. The API will return the predicted category (Science, Business, Health, or Entertainment).

### Example Request

```json
{
  "title": "New Breakthrough in Quantum Computing"
}
```

### Example Response

```json
{
  "category": "Science"
}
```

## Technologies Used

- **Hugging Face Transformers**: For model selection and fine-tuning.
- **DistilBERT**: The transformer model used as the base for fine-tuning.
- **AWS Lambda**: For serverless deployment of the model.
- **AWS API Gateway**: For creating a REST API interface to interact with the model.

## Results

The model achieved high accuracy on the test set and performed well in categorizing titles across all four categories. The load testing indicated that AWS Lambda was able to handle thousands of requests with minimal latency, making it a suitable choice for serverless deployment.

## Future Improvements

- **Additional Categories**: Expand the classification to include more categories.
- **Model Optimization**: Further fine-tune the model to improve accuracy and reduce inference time.
- **User Interface**: Develop a simple UI to make interacting with the model more user-friendly.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.
