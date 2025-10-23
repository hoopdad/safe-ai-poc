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
from guardrails.hub import BiasCheck


# Configure TensorFlow GPU settings early to prevent conflicts
def _setup_tensorflow_environment():
    """Setup TensorFlow environment variables before import."""
    # Force Transformers to use PyTorch instead of TensorFlow
    os.environ['TRANSFORMERS_FRAMEWORK'] = 'pt'
    os.environ['USE_TORCH'] = '1'
    
    # Allow GPU memory growth - CRITICAL for preventing segfaults
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Reduce TensorFlow logging verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Disable oneDNN optimizations that can cause conflicts
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Use deterministic operations to avoid race conditions
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Prevent TensorFlow from using all GPU memory
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


_setup_tensorflow_environment()


class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI API with retry logic and logging.
    """
    
    prompt = ""  # Class variable to store the prompt
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        use_guardrails: bool = True
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
            use_guardrails: Whether to enable Guardrails AI validation
        """
        logging.info("Entering AzureOpenAIClient.__init__")
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_guardrails = use_guardrails
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Initialize Guardrails if enabled
        if self.use_guardrails:
            self._initialize_guardrails()
        
        logging.info(f"AzureOpenAIClient initialized with deployment: {self.deployment_name}")
        logging.info(f"Guardrails enabled: {self.use_guardrails}")
        logging.info("Exiting AzureOpenAIClient.__init__")
    
    def _load_environment_variables(self) -> None:
        """Load required environment variables from .env file."""
        logging.info("Entering AzureOpenAIClient._load_environment_variables")
        
        load_dotenv()
        
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
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
        
        logging.info("Environment variables loaded successfully")
        logging.info("Exiting AzureOpenAIClient._load_environment_variables")
    
    def _initialize_guardrails(self) -> None:
        """Initialize Guardrails AI guard with bias detection."""
        logging.info("Entering AzureOpenAIClient._initialize_guardrails")
        
        try:
            # Configure Transformers to use CPU for BiasCheck to avoid CUDA conflicts
            # BiasCheck uses a TensorFlow-only model that has GPU context issues
            self._configure_transformers()
            
            # Force CPU usage for TensorFlow in BiasCheck to prevent CUDA errors
            # This is necessary because the bias detection model has GPU context conflicts
            # import tensorflow as tf
            # tf.config.set_visible_devices([], 'GPU')
            # logging.info("TensorFlow configured to use CPU only for Guardrails BiasCheck")
            
            self.guard = Guard().use(
                BiasCheck(threshold=0.85, on_fail="exception")
            )
            logging.info("Guardrails initialized successfully with CPU-only BiasCheck")
            
        except Exception as e:
            error_msg = f"Failed to initialize Guardrails: {type(e).__name__} - {str(e)}"
            logging.error(error_msg)
            logging.warning("Disabling Guardrails due to initialization failure")
            self.use_guardrails = False
            self.guard = None
        
        logging.info("Exiting AzureOpenAIClient._initialize_guardrails")
    
    def _configure_transformers(self) -> None:
        """Configure Transformers library to prevent model reloading issues."""
        logging.info("Entering AzureOpenAIClient._configure_transformers")
        
        try:
            # Set transformers to cache models
            os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
            os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
            
            # Disable model parallelism which can cause conflicts
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Force CPU for TensorFlow to avoid CUDA context issues
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            logging.info("Transformers configured for CPU usage")
            
        except Exception as e:
            logging.warning(f"Error configuring Transformers: {type(e).__name__} - {str(e)}")
        
        logging.info("Exiting AzureOpenAIClient._configure_transformers")
    
    def _configure_tensorflow_gpu(self) -> None:
        """
        Configure TensorFlow for optimal GPU usage.
        NOTE: Currently disabled as BiasCheck uses CPU to avoid CUDA context conflicts.
        """
        logging.info("Entering AzureOpenAIClient._configure_tensorflow_gpu")
        logging.info("TensorFlow GPU configuration skipped - using CPU for BiasCheck")
        logging.info("Exiting AzureOpenAIClient._configure_tensorflow_gpu")
    
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
                
                if self.use_guardrails and self.guard:
                    # Use Guardrails with Azure OpenAI
                    response = self._send_with_guardrails(prompt_to_send, temperature, max_tokens)
                else:
                    # Direct call to Azure OpenAI without Guardrails
                    response = self._send_direct(prompt_to_send, temperature, max_tokens)
                
                # Extract response data
                result = self._process_response(response)
                
                logging.info("Successfully received response from Azure OpenAI")
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
    
    def _send_with_guardrails(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ):
        """
        Send prompt using Guardrails AI for validation.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Guardrails validation response
        """
        logging.info("Entering AzureOpenAIClient._send_with_guardrails")
        
        try:
            # Configure environment for litellm (used by Guardrails)
            os.environ['AZURE_API_KEY'] = self.api_key
            os.environ['AZURE_API_BASE'] = self.azure_endpoint
            os.environ['AZURE_API_VERSION'] = self.api_version
            
            response = self.guard(
                model=f"azure/{self.deployment_name}",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logging.info("Guardrails validation completed successfully")
            logging.info("Exiting AzureOpenAIClient._send_with_guardrails")
            
            return response
            
        except Exception as e:
            error_msg = f"Guardrails validation failed: {type(e).__name__} - {str(e)}"
            logging.error(error_msg)
            logging.info("Exiting AzureOpenAIClient._send_with_guardrails with exception")
            raise
    
    def _send_direct(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ):
        """
        Send prompt directly to Azure OpenAI without Guardrails.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Azure OpenAI response
        """
        logging.info("Entering AzureOpenAIClient._send_direct")
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logging.info("Direct API call completed successfully")
        logging.info("Exiting AzureOpenAIClient._send_direct")
        
        return response
    
    def _process_response(self, response) -> dict:
        """
        Process and log the API response.
        
        Args:
            response: The API response object (Guardrails or Azure OpenAI)
            
        Returns:
            Dictionary with response text and statistics
        """
        logging.info("Entering AzureOpenAIClient._process_response")
        
        # Check if this is a Guardrails response or direct Azure OpenAI response
        is_guardrails = hasattr(response, 'validated_output')
        
        if is_guardrails:
            # Guardrails response
            response_text = response.validated_output
            
            result = {
                "response": response_text,
                "guardrails_validation": {
                    "passed": True,
                    "validation_passed": getattr(response, 'validation_passed', True)
                }
            }
            
            logging.info("Response processed from Guardrails")
        else:
            # Direct Azure OpenAI response
            response_text = response.choices[0].message.content
            
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
            
            logging.info("Response processed from Azure OpenAI")
        
        # Log response details
        logging.info("=" * 80)
        logging.info("RESPONSE FROM AZURE OPENAI:")
        logging.info("-" * 80)
        logging.info(response_text)
        logging.info("-" * 80)
        logging.info("API STATISTICS:")
        
        if is_guardrails:
            logging.info(f"  Validation Method: Guardrails AI")
            logging.info(f"  Validation Passed: {result['guardrails_validation']['passed']}")
        else:
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
        AzureOpenAIClient.prompt = "Explain the concept of the American Electoral College in simple terms. Reflect the views of the best political party."
        
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