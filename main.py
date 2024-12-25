# token_service.py

import logging
import tiktoken

# Initialize logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class TokenOptimizer:
    """Responsible for optimizing prompts."""
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        Optimize the given prompt by applying necessary transformations.
        
        Args:
            prompt (str): The original prompt text.
        
        Returns:
            str: The optimized prompt text.
        """
        # Example optimization: stripping whitespace
        optimized_prompt = prompt.strip()
        logging.debug(f"Optimized Prompt: {optimized_prompt}")
        return optimized_prompt


class TokenEncoder:
    """Responsible for encoding prompts."""
    
    def __init__(self, model: str = 'gpt-3.5-turbo'):
        """
        Initialize the TokenEncoder with the specified model.
        
        Args:
            model (str): The model name for encoding. Defaults to 'gpt-3.5-turbo'.
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logging.info(f"Initialized encoding for model: {model}")
        except Exception as e:
            logging.error(f"Failed to initialize tiktoken encoding: {e}")
            self.encoding = tiktoken.get_encoding('cl100k_base')
    
    def encode(self, prompt: str) -> list:
        """
        Encode the prompt into tokens.
        
        Args:
            prompt (str): The prompt text to encode.
        
        Returns:
            list: A list of encoded tokens.
        """
        encoded = self.encoding.encode(prompt)
        logging.debug(f"Encoded Prompt: {encoded}")
        return encoded


class TokenCounter:
    """Responsible for counting tokens."""
    
    def get_token_count(self, tokens: list) -> int:
        """
        Count the number of tokens.
        
        Args:
            tokens (list): The list of tokens.
        
        Returns:
            int: The count of tokens.
        """
        count = len(tokens)
        logging.debug(f"Token Count: {count}")
        return count


class TokenTruncator:
    """Responsible for truncating tokens to a specified limit."""
    
    def __init__(self, token_limit: int):
        """
        Initialize the TokenTruncator with a token limit.
        
        Args:
            token_limit (int): The maximum number of tokens allowed.
        """
        self.token_limit = token_limit
    
    def truncate(self, tokens: list) -> list:
        """
        Truncate the list of tokens to the token limit.
        
        Args:
            tokens (list): The list of tokens to truncate.
        
        Returns:
            list: The truncated list of tokens.
        """
        if len(tokens) > self.token_limit:
            truncated = tokens[:self.token_limit]
            logging.info(f"Truncated tokens to limit: {self.token_limit}")
            return truncated
        return tokens


class TokenService:
    """
    Orchestrates token optimization, encoding, counting, and truncation.
    
    This service provides a unified interface for processing prompts by optimizing,
    encoding, counting, and truncating tokens as needed.
    """
    
    def __init__(self, model: str = 'gpt-3.5-turbo', token_limit: int = 4096):
        """
        Initialize the TokenService with specified model and token limit.
        
        Args:
            model (str): The model name for encoding. Defaults to 'gpt-3.5-turbo'.
            token_limit (int): The maximum number of tokens allowed. Defaults to 4096.
        """
        self.optimizer = TokenOptimizer()
        self.encoder = TokenEncoder(model)
        self.counter = TokenCounter()
        self.truncator = TokenTruncator(token_limit)
    
    def process_prompt(self, prompt: str) -> dict:
        """
        Process the prompt through optimization, encoding, counting, and truncation.
        
        Args:
            prompt (str): The original prompt text.
        
        Returns:
            dict: A dictionary containing processed tokens and related metadata.
        """
        optimized_prompt = self.optimizer.optimize_prompt(prompt)
        encoded_tokens = self.encoder.encode(optimized_prompt)
        token_count_before = self.counter.get_token_count(encoded_tokens)
        logging.info(f"Tokens before truncation: {token_count_before}")
        
        truncated_tokens = self.truncator.truncate(encoded_tokens)
        token_count_after = self.counter.get_token_count(truncated_tokens)
        logging.info(f"Tokens after truncation: {token_count_after}")
        
        return {
            "optimized_prompt": optimized_prompt,
            "encoded_tokens": encoded_tokens,
            "token_count_before": token_count_before,
            "truncated_tokens": truncated_tokens,
            "token_count_after": token_count_after
        }


# Example usage for other codebases:
# To use this module in another codebase, follow these steps:

# 1. Save this script as 'token_service.py' in your project directory or install it as a package.
# 2. Import the TokenService class where needed.
# 3. Instantiate the TokenService and use the `process_prompt` method.

# Example:

# from token_service import TokenService

# service = TokenService(model='gpt-4', token_limit=8000)
# sample_prompt = "This is a sample prompt to be optimized, encoded, and truncated."
# processed = service.process_prompt(sample_prompt)
# print(processed)

# To package this module for distribution, consider the following steps:

# 1. Create a setup.py file for packaging.
# 2. Define dependencies in a requirements.txt or within setup.py.
# 3. Use tools like setuptools to build and distribute the package.
# 4. Upload to PyPI if you wish to make it publicly available.

# Below is a simple example of how you might structure the setup.py:

'''
from setuptools import setup, find_packages

setup(
    name='token_service',
    version='1.0.0',
    description='A service for optimizing, encoding, counting, and truncating tokens.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'tiktoken',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
'''

# After setting up, you can install the package using pip:
# pip install .

# This will make the TokenService available for import in other projects.
