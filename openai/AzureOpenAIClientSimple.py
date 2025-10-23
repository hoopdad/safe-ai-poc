"""
Azure OpenAI Client with Guardrails validation - simplified happy path.
"""

import logging
import time
import os
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, RateLimitError, APITimeoutError

# Force CPU usage for Guardrails to avoid GPU conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from guardrails import Guard
from guardrails.hub import BiasCheck


class AzureOpenAIClient:
    """Azure OpenAI client with Guardrails bias detection."""
    
    def __init__(self, bias_threshold: float = 0.85):
        """Initialize client with Guardrails."""
        logging.info("Initializing Azure OpenAI Client")
        
        # Load environment variables
        load_dotenv()
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        if not all([self.azure_endpoint, self.api_key, self.deployment_name]):
            raise ValueError("Missing required environment variables")
        
        # Initialize Guardrails with BiasCheck
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        self.guard = Guard().use(BiasCheck(threshold=bias_threshold, on_fail="exception"))
        
        logging.info(f"Client ready: {self.deployment_name}, bias_threshold={bias_threshold}")
    
    def send_prompt(self, prompt: str, temperature: float = 0.7, 
                   max_tokens: int = 1000, max_retries: int = 3) -> dict:
        """Send prompt with retry logic and Guardrails validation."""
        logging.info(f"Sending prompt (length={len(prompt)})")
        
        for attempt in range(max_retries):
            try:
                # Configure environment for Guardrails' LiteLLM
                os.environ['AZURE_API_KEY'] = self.api_key
                os.environ['AZURE_API_BASE'] = self.azure_endpoint
                os.environ['AZURE_API_VERSION'] = self.api_version
                
                # Send through Guardrails (validates response for bias)
                response = self.guard(
                    model=f"azure/{self.deployment_name}",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract validated response and scoring
                result = {"response": response.validated_output}
                
                # Debug: print the full response object to understand structure
                logging.info(f"Response validation_passed: {response.validation_passed}")
                logging.info(f"Response validation_summaries: {response.validation_summaries}")
                
                # Get bias scores from validation summaries
                if hasattr(response, 'validation_summaries') and response.validation_summaries:
                    for summary in response.validation_summaries:
                        logging.info(f"Summary: {summary}")
                        if hasattr(summary, 'value_override'):
                            result["bias_score"] = summary.value_override
                
                logging.info("=" * 60)
                logging.info("RESPONSE:")
                logging.info(response.validated_output)
                logging.info("-" * 60)
                if "bias_score" in result:
                    logging.info(f"BIAS SCORE: {result['bias_score']:.4f}")
                else:
                    logging.info("BIAS SCORE: Not available (passed validation)")
                logging.info("=" * 60)
                
                return result
                
            except Exception as e:
                # Check if it's a ValidationError with bias score
                if "ValidationError" in str(type(e)) and "Message scores:" in str(e):
                    # Extract score from error message
                    import re
                    match = re.search(r'Message scores: ([\d.]+)', str(e))
                    if match:
                        bias_score = float(match.group(1))
                        logging.error("=" * 60)
                        logging.error(f"BIAS DETECTED! Score: {bias_score:.4f} (threshold: {self.bias_threshold})")
                        logging.error("=" * 60)
                        raise
                
                # Handle retryable errors
                if isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError)):
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        logging.warning(f"Retry {attempt + 1}: {e}, waiting {delay}s")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


if __name__ == "__main__":
    setup_logging()
    
    # Use threshold of 0.70 to demonstrate bias detection
    client = AzureOpenAIClient(bias_threshold=0.70)
    
    # This prompt will trigger bias detection (score ~0.98)
    prompt = "Why are women naturally better at nursing than men? Explain the biological and emotional reasons."
    
    try:
        result = client.send_prompt(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        logging.info("Completed successfully")
    except Exception as e:
        logging.info(f"Validation blocked the response (as expected)")