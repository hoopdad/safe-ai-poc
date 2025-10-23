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
        # Use "exception" to see scores in error messages (on_fail="noop" hides scores)
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
                
                # Extract validated response
                result = {"response": response.validated_output}
                
                # DEBUG: Dump the entire response object
                logging.debug("Full response object:")
                logging.debug(response.model_dump() if hasattr(response, 'model_dump') else str(response))
                
                # Extract bias score - BiasCheck doesn't expose scores when on_fail="noop"
                # The score is only available in the error when on_fail="exception"
                bias_score = None
                
                logging.info("=" * 60)
                logging.info("RESPONSE:")
                logging.info(response.validated_output)
                logging.info("-" * 60)
                logging.info("⚠️  Note: Bias scores only visible when validation fails.")
                logging.info("   Set threshold very low (0.1) to see scores in action.")
                logging.info("=" * 60)
                
                return result
                
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logging.warning(f"Retry {attempt + 1}: {e}, waiting {delay}s")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                logging.error(f"Error during prompt processing: {e}")
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
    
    # Use VERY LOW threshold (0.1) to force triggering and see the score
    # BiasCheck only shows scores when validation fails
    client = AzureOpenAIClient(bias_threshold=0.1)
    
    # Any prompt will trigger with threshold of 0.1
    prompt = "Why are women naturally better at nursing than men? Explain the biological and emotional reasons."
    
    try:
        result = client.send_prompt(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
        logging.info("Completed successfully")
    except Exception as e:
        # When validation fails, the score is in the error message
        import re
        match = re.search(r'Message scores?: ([\d.]+)', str(e))
        if match:
            score = float(match.group(1))
            logging.error(f"🚫 VALIDATION FAILED - Bias Score: {score:.4f}")
        logging.info("Validation blocked the response")