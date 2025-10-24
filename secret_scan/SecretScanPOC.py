"""
Azure OpenAI Client with Guardrails - scanning INPUT before sending.
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
from guardrails.hub import SecretsPresent, DetectPII


class AzureOpenAIClient:
    """Azure OpenAI client with INPUT validation for secrets/PII."""
    
    def __init__(self):
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
        
        # Initialize Azure OpenAI client directly (not through Guardrails)
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Initialize Guardrails for INPUT validation (prompt scanning)
        # on_fail="fix" will automatically redact secrets/PII instead of blocking
        self.input_guard = Guard().use_many(
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
                on_fail="fix"  # Auto-redact PII
            ),
            SecretsPresent(on_fail="fix")  # Auto-redact secrets
        )
        
        logging.info(f"Client ready: {self.deployment_name}")
        logging.info("Input validation enabled: SecretsPresent, DetectPII")
        logging.info("Mode: AUTO-REDACT (secrets/PII will be replaced with ***)")
    
    def send_prompt(self, prompt: str, temperature: float = 0.7, 
                   max_tokens: int = 1000, max_retries: int = 3) -> dict:
        """Send prompt with INPUT validation and retry logic."""
        logging.info(f"📤 Preparing prompt (length={len(prompt)})")
        
        # STEP 1: Validate INPUT locally BEFORE sending
        logging.info("=" * 60)
        logging.info("🔍 SCANNING INPUT for secrets/PII...")
        logging.info("   (This happens LOCALLY - no API call yet)")
        logging.info("=" * 60)
        
        try:
            # Validate and auto-redact the prompt text
            validated = self.input_guard.parse(prompt)
            cleaned_prompt = validated.validated_output
            
            if cleaned_prompt != prompt:
                logging.warning("⚠️  SECRETS/PII DETECTED - Auto-redacted!")
                logging.info(f"   Original length: {len(prompt)} chars")
                logging.info(f"   Cleaned length: {len(cleaned_prompt)} chars")
                logging.info("   Changes:")
                # Show what was redacted
                import difflib
                diff = list(difflib.unified_diff(
                    prompt.splitlines(keepends=True),
                    cleaned_prompt.splitlines(keepends=True),
                    lineterm='',
                    n=0
                ))
                for line in diff[2:6]:  # Show first few diff lines
                    if line.startswith('-') or line.startswith('+'):
                        logging.info(f"     {line[:80]}")
            else:
                logging.info("✅ INPUT CLEAN - No secrets/PII detected")
            
            # Use the cleaned prompt for LLM
            prompt_to_send = cleaned_prompt
            
        except Exception as e:
            logging.error("=" * 60)
            logging.error("🚫 INPUT VALIDATION ERROR")
            logging.error(f"   Error: {str(e)[:200]}")
            logging.error("=" * 60)
            raise ValueError(f"Input validation failed: {e}")
        
        # STEP 2: Send to LLM (only if input validation passed)
        for attempt in range(max_retries):
            try:
                logging.info("=" * 60)
                logging.info("🚀 SENDING TO AZURE OPENAI...")
                logging.info("=" * 60)
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt_to_send}],  # Use cleaned prompt
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                result = {
                    "response": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
                logging.info("=" * 60)
                logging.info("✅ RESPONSE RECEIVED")
                logging.info(f"   Tokens used: {result['usage']['total_tokens']}")
                logging.info("=" * 60)
                logging.info("RESPONSE:")
                logging.info(result['response'])
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
                logging.error(f"❌ LLM Error: {e}")
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
    
    client = AzureOpenAIClient()
    
    # Test prompts - secrets/PII will be AUTO-REDACTED
    prompts = [
        {
            "name": "Prompt with API key (will be redacted)",
            "text": "Review this code:\nimport os\nif __name__ == '__main__':\n    mykey = 'sk-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz'\n    print(mykey)"
        },
        {
            "name": "Prompt with git commit hash (false positive - will be redacted)",
            "text": "Review this git commit:\ncommit 4a5f6e8d9c1b2a3e4f5a6b7c8d9e0f1a2b3c4d5e\nAuthor: Developer\nDate: 2024-10-20\n\nFixed bug in authentication"
        },
        {
            "name": "Prompt with email (will be redacted)",
            "text": "Review this code:\nimport os\n# contact codeguy@example.com with questions\nif __name__ == '__main__':\n    print('Hello World')"
        },
        {
            "name": "Clean prompt (no changes)",
            "text": "Explain what this code does:\nimport os\nif __name__ == '__main__':\n    print('Hello World')"
        }
    ]
    
    for test in prompts:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f"TEST: {test['name']}")
        logging.info("=" * 80)
        
        try:
            result = client.send_prompt(
                prompt=test['text'],
                temperature=0.7,
                max_tokens=150
            )
            logging.info(f"✅ SUCCESS - Prompt sent (possibly redacted)")
            
        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}")
        
        time.sleep(1)  # Brief pause between tests
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("Testing completed")
    logging.info("=" * 80)