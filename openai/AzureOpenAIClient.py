"""
Azure OpenAI Client with retry logic and comprehensive logging.
"""

import logging
import time
import os
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from guardrails import Guard


class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI API with retry logic and logging.
    """
    
    prompt = ""  # Class variable to store the prompt
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
        """
        logging.info("Entering AzureOpenAIClient.__init__")
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        logging.info(f"AzureOpenAIClient initialized with deployment: {self.deployment_name}")
        logging.info("Exiting AzureOpenAIClient.__init__")
    
    def _load_environment_variables(self) -> None:
        """Load required environment variables from .env file."""
        logging.info("Entering AzureOpenAIClient._load_environment_variables")
        
        load_dotenv()
        
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        self.guardrails_ai_key = os.getenv("GUARDRAILS_AI_KEY")
        
        # Validate required environment variables
        if not self.azure_endpoint:
            error_msg = "AZURE_OPENAI_ENDPOINT environment variable is not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.api_key:
            error_msg = "AZURE_OPENAI_API_KEY environment variable is not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.deployment_name:
            error_msg = "AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.guardrails_ai_key:
            error_msg = "GUARDRAILS_AI_KEY environment variable is not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info("Environment variables loaded successfully")
        logging.info("Exiting AzureOpenAIClient._load_environment_variables")
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current retry attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        logging.info(f"Entering AzureOpenAIClient._calculate_delay with attempt={attempt}")
        
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        logging.info(f"Calculated delay: {delay} seconds")
        logging.info("Exiting AzureOpenAIClient._calculate_delay")
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        logging.info("Entering AzureOpenAIClient._should_retry")
        
        retryable_exceptions = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError
        )
        
        should_retry = isinstance(exception, retryable_exceptions)
        
        logging.info(f"Exception type: {type(exception).__name__}, Should retry: {should_retry}")
        logging.info("Exiting AzureOpenAIClient._should_retry")
        
        return should_retry
    
    def send_prompt(
        self,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> dict:
        """
        Send a prompt to Azure OpenAI with retry logic.
        
        Args:
            prompt: The prompt to send (uses class variable if not provided)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing response and statistics
        """
        logging.info("Entering AzureOpenAIClient.send_prompt")
        
        # Use provided prompt or fall back to class variable
        prompt_to_send = prompt if prompt is not None else self.prompt
        
        if not prompt_to_send:
            error_msg = "No prompt provided and class variable 'prompt' is empty"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info(f"Prompt length: {len(prompt_to_send)} characters")
        logging.info(f"Parameters - temperature: {temperature}, max_tokens: {max_tokens}")
        
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                logging.info(f"Attempt {attempt + 1} of {self.max_retries}")
                
                guard = Guard()

                # response = guard(
                #     model="gpt-3.5-turbo",
                #     messages=[{
                #         "role": "user",
                #         "content": prompt_to_send
                #     }]
                # )
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "user", "content": prompt_to_send}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract response data
                result = self._process_response(response)
                logging.info("Successfully received response from Azure OpenAI")

                if (response.choices[0].message.content):
                    result = guard.parse(
                        llm_output=response.choices[0].message.content
                    )
                logging.info("Successfully received response from Guardrails AI")
                
                logging.info("Exiting AzureOpenAIClient.send_prompt")
                return result
                
            except Exception as e:
                last_exception = e
                error_msg = f"Error on attempt {attempt + 1}: {type(e).__name__} - {str(e)}"
                logging.error(error_msg)
                
                if self._should_retry(e) and attempt < self.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    logging.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    # Log and re-raise if can't be handled gracefully
                    logging.error(f"All retry attempts exhausted or non-retryable error encountered")
                    logging.error(f"Final error: {type(e).__name__} - {str(e)}")
                    logging.info("Exiting AzureOpenAIClient.send_prompt with exception")
                    raise
        
        # This should not be reached, but included for safety
        if last_exception:
            logging.error("Max retries reached without success")
            logging.info("Exiting AzureOpenAIClient.send_prompt with exception")
            raise last_exception
    
    def _process_response(self, response) -> dict:
        """
        Process and log the API response.
        
        Args:
            response: The API response object
            
        Returns:
            Dictionary with response text and statistics
        """
        logging.info("Entering AzureOpenAIClient._process_response")
        
        # Extract response content
        response_text = response.choices[0].message.content
        
        # Extract statistics
        result = {
            "response": response_text,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
        
        # Log response details
        logging.info("=" * 80)
        logging.info("RESPONSE FROM AZURE OPENAI:")
        logging.info("-" * 80)
        logging.info(response_text)
        logging.info("-" * 80)
        logging.info("API STATISTICS:")
        logging.info(f"  Model: {result['model']}")
        logging.info(f"  Prompt Tokens: {result['usage']['prompt_tokens']}")
        logging.info(f"  Completion Tokens: {result['usage']['completion_tokens']}")
        logging.info(f"  Total Tokens: {result['usage']['total_tokens']}")
        logging.info(f"  Finish Reason: {result['finish_reason']}")
        logging.info("=" * 80)
        
        logging.info("Exiting AzureOpenAIClient._process_response")
        
        return result


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging(logging.INFO)
    
    logging.info("Starting Azure OpenAI Client application")
    
    try:
        # Set the class variable prompt
        AzureOpenAIClient.prompt = "Explain the concept of recursion in programming in simple terms."
        
        # Initialize client
        client = AzureOpenAIClient(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0
        )
        
        # Send prompt and get response
        result = client.send_prompt(temperature=0.7, max_tokens=500)
        
        logging.info("Application completed successfully")
        
    except Exception as e:
        logging.error(f"Application failed with error: {type(e).__name__} - {str(e)}")
        raise
    
    logging.info("Azure OpenAI Client application finished")
