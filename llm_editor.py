#!/usr/bin/env python3
import os
import re
import sys
import json
import base64
import logging

import requests
from typing import Tuple, Optional, Dict, Any
from bs4 import BeautifulSoup
import google.generativeai as genai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_editor")

# GitHub API URLs
GITHUB_API = "https://api.github.com"
REPO_OWNER = os.environ.get("REPO_OWNER")
REPO_NAME = os.environ.get("REPO_NAME")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini API URL
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

class GitHubAPIError(Exception):
    """Exception raised for GitHub API errors"""
    pass

class GeminiAPIError(Exception):
    """Exception raised for Gemini API errors"""
    pass

class HTMLExtractionError(Exception):
    """Exception raised for HTML extraction errors"""
    pass

def github_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make a request to the GitHub API
    
    Args:
        method: HTTP method (GET, POST, PUT, PATCH)
        endpoint: API endpoint
        data: Request data (for POST, PUT, PATCH)
    
    Returns:
        Response data as dictionary
    """
    url = f"{GITHUB_API}/{endpoint}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json() if response.content else {}
    
    except requests.RequestException as e:
        raise GitHubAPIError(f"GitHub API request failed: {str(e)}")

def get_issue_details() -> Dict[str, Any]:
    """
    Get the details of the current issue
    
    Returns:
        Issue data as dictionary
    """
    endpoint = f"repos/{REPO_OWNER}/{REPO_NAME}/issues/{ISSUE_NUMBER}"
    return github_request("GET", endpoint)

def get_current_html() -> Tuple[str, str]:
    """
    Get the current content of index.html
    
    Returns:
        Tuple of (file content, sha)
    """
    endpoint = f"repos/{REPO_OWNER}/{REPO_NAME}/contents/index.html"
    response = github_request("GET", endpoint)
    
    if "content" not in response:
        raise GitHubAPIError("Could not retrieve index.html content")
    
    content = base64.b64decode(response["content"]).decode("utf-8")
    return content, response["sha"]

def extract_html_from_llm_response(response_text: str) -> str:
    """
    Extract HTML code from the LLM response
    
    Args:
        response_text: Raw response from the LLM
    
    Returns:
        Extracted HTML content
    """
    # Try to extract code between ```html and ``` markers
    html_pattern = re.compile(r'```(?:html)?\s*([\s\S]*?)\s*```')
    matches = html_pattern.findall(response_text)
    
    if matches:
        return matches[0].strip()
    
    # If no matches with markers, try to extract anything that looks like HTML
    try:
        soup = BeautifulSoup(response_text, 'html.parser')
        html_tags = soup.find_all(['html', 'body', 'div', 'section', 'header', 'footer', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if html_tags:
            # Return the entire text as it might be valid HTML
            return response_text.strip()
    except Exception as e:
        logger.warning(f"Error parsing HTML with BeautifulSoup: {str(e)}")
    
    # If we got here, we couldn't extract anything that looks like HTML
    raise HTMLExtractionError("Could not extract HTML from LLM response")


def add_form_back(form_html: str, new_html: str) -> str:
    """
    Add the form back into the HTML content
    
    Args:
        form_html: The form HTML to add back
        new_html: The new HTML content
    
    Returns:
        HTML with form added at the top of the body
    """
    soup = BeautifulSoup(new_html, 'html.parser')
    form_soup = BeautifulSoup(form_html, 'html.parser')
    
    # Get the form element from the soup
    form_element = form_soup.find() if form_soup.find() else form_soup
    
    # Find the body tag
    body_tag = soup.find('body')
    
    if body_tag:
        # Insert the form at the beginning of the body
        body_tag.insert(0, form_element)
    else:
        # No body tag? Create one
        html_tag = soup.find('html')
        if html_tag:
            body_tag = soup.new_tag('body')
            body_tag.append(form_element)
            html_tag.append(body_tag)
        else:
            # No HTML structure at all? Create a basic one
            soup = BeautifulSoup('<html><body></body></html>', 'html.parser')
            soup.body.append(form_element)
            
            # Try to extract content from new HTML and add after form
            content_soup = BeautifulSoup(new_html, 'html.parser')
            for tag in content_soup.find_all(['div', 'section', 'p', 'h1', 'h2', 'h3', 'script', 'style']):
                soup.body.append(tag)
    
    return str(soup)

def extract_form(html_content: str) -> Tuple[str, str]:
    """
    Extract the form from the HTML content
    
    Args:
        html_content: HTML content
    
    Returns:
        Tuple of (form HTML, HTML content without form)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for a form element first
    form = soup.find('form')
    
    # If no form is found, look for input+button combination
    if not form:
        inputs = soup.find_all('input', {'type': 'text'})
        buttons = soup.find_all('button')
        
        if inputs and buttons:
            # Find common parent that contains both input and button
            for input_tag in inputs:
                parent = input_tag.parent
                if parent.find('button'):
                    form = parent
                    break
    
    if not form:
        # If still no form is found, create a default one
        default_form = """
        <div id="prompt-form" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>Modify this webpage</h3>
            <form action="https://github.com/REPO_OWNER/REPO_NAME/issues/new" method="get" target="_blank">
                <input type="hidden" name="labels" value="prompt">
                <input type="text" name="body" placeholder="Enter your instructions..." style="width: 70%; padding: 8px; margin-right: 10px;">
                <button type="submit" style="padding: 8px 15px; background-color: #0366d6; color: white; border: none; border-radius: 3px; cursor: pointer;">Submit</button>
            </form>
            <p><small>Your request will be processed through GitHub Issues</small></p>
        </div>
        """
        return default_form.replace("REPO_OWNER", REPO_OWNER).replace("REPO_NAME", REPO_NAME), html_content
    
    # Extract the form HTML
    form_html = str(form)
    
    # Remove the form from the HTML
    form.extract()
    
    return form_html, str(soup)

def get_gemini_completion(prompt: str) -> str:
    """
    Get a completion from the Gemini API using Google's genai library
    
    Args:
        prompt: The prompt to send to the API
    
    Returns:
        The model's response
    """
    try:
        # Configure the client
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create a chat session with the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Send the prompt and get the response
        response = model.generate_content(prompt)
        
        # Check if we have a valid response
        if response and response.text:
            return response.text
        else:
            raise GeminiAPIError("No completion returned from Gemini API")
    
    except Exception as e:
        raise GeminiAPIError(f"Gemini API request failed: {str(e)}")

def create_prompt(html_content: str, user_instructions: str) -> str:
    """
    Create a prompt for the LLM
    
    Args:
        html_content: Current HTML content (without the form)
        user_instructions: Instructions from the user
    
    Returns:
        Formatted prompt for the LLM
    """
    return f"""
Please modify the following HTML/CSS/JS webpage according to these instructions:

INSTRUCTIONS: {user_instructions}

IMPORTANT RULES:
1. Return the COMPLETE updated HTML file, not just a snippet or the changes.
2. Respond with only the HTML code, no explanations or markdown formatting.
3. Feel free to enhance the CSS and add JavaScript as needed.
4. Maintain the overall structure with html, head, and body tags.

CURRENT HTML:
```html
{html_content}
```

Provide the complete updated HTML file:
"""

def commit_file(file_content: str, file_sha: str) -> str:
    """
    Commit the updated index.html file
    
    Args:
        file_content: New content for the file
        file_sha: Current SHA of the file
    
    Returns:
        Commit SHA
    """
    endpoint = f"repos/{REPO_OWNER}/{REPO_NAME}/contents/index.html"
    commit_message = f"Update index.html based on issue #{ISSUE_NUMBER}"
    
    data = {
        "message": commit_message,
        "content": base64.b64encode(file_content.encode("utf-8")).decode("utf-8"),
        "sha": file_sha,
        "branch": "main"
    }
    
    response = github_request("PUT", endpoint, data)
    
    if "commit" not in response or "sha" not in response["commit"]:
        raise GitHubAPIError("Failed to commit changes")
    
    return response["commit"]["sha"]



def add_comment_to_issue(comment: str) -> None:
    """
    Add a comment to the GitHub issue
    
    Args:
        comment: Comment content
    """
    endpoint = f"repos/{REPO_OWNER}/{REPO_NAME}/issues/{ISSUE_NUMBER}/comments"
    data = {"body": comment}
    github_request("POST", endpoint, data)

def main() -> int:
    """
    Main function to process a website modification request
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Get issue details
        issue = get_issue_details()
        user_instructions = issue["body"]
        logger.info(f"Processing issue #{ISSUE_NUMBER}: {issue['title']}")
        
        # Get current HTML content
        current_html, file_sha = get_current_html()
        logger.info("Retrieved current index.html content")
        
        # Extract the form and get HTML without it
        form_html, html_without_form = extract_form(current_html)
        logger.info("Extracted form from HTML")
        
        # Create prompt for Gemini
        prompt = create_prompt(html_without_form, user_instructions)
        logger.info("Created prompt for LLM (without form)")
        
        # Get completion from Gemini
        llm_response = get_gemini_completion(prompt)
        logger.info("Received response from Gemini")
        
        # Extract HTML from the response
        try:
            new_html = extract_html_from_llm_response(llm_response)
            logger.info("Extracted HTML from LLM response")
        except HTMLExtractionError:
            # If HTML extraction fails, use the raw response
            logger.warning("HTML extraction failed, using raw LLM response")
            new_html = llm_response
        
        # Add form back to the new HTML
        final_html = add_form_back(form_html, new_html)
        logger.info("Added form back to the modified HTML")
        
        # Commit the changes
        commit_sha = commit_file(final_html, file_sha)
        logger.info(f"Committed changes with SHA: {commit_sha}")
        
        # Create commit URL
        commit_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/commit/{commit_sha}"
        
        # Create comment
        comment = f"""
## Website Updated! 🎉

Your changes have been applied to the website based on your instructions.

### 🔗 Links
- [View Commit]({commit_url})
- [View Website](https://{REPO_OWNER}.github.io/{REPO_NAME}/)

If you need further adjustments, please create a new issue with the 'prompt' label.
"""
        
        # Add comment to issue
        add_comment_to_issue(comment)
        logger.info("Added comment to issue")
        
        # Close the issue
        endpoint = f"repos/{REPO_OWNER}/{REPO_NAME}/issues/{ISSUE_NUMBER}"
        github_request("PATCH", endpoint, {"state": "closed"})
        logger.info("Closed issue")
        
        return 0
    
    except (GitHubAPIError, GeminiAPIError, HTMLExtractionError) as e:
        error_message = f"""
## ❌ Error Updating Website

Sorry, an error occurred while processing your request:

```
{str(e)}
```

Please check your instructions and try again. If the problem persists, contact the repository maintainer.
"""
        try:
            add_comment_to_issue(error_message)
            logger.error(f"Failed to update website: {str(e)}")
        except Exception as comment_error:
            logger.error(f"Failed to add error comment: {str(comment_error)}")
            
        return 1

if __name__ == "__main__":
    sys.exit(main())