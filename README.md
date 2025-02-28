# Prompt Place

Inspired by reddit [r/place](https://www.reddit.com/r/place/) Prompt Place enables any github user to modify `index.html` by submitting a natural language change requests. 

Start by [clicking here to create an issue with the Prompt template](https://github.com/mnaei/PromptPlace2/issues/new?template=prompt.md) and outline instructions for the desired modification in the issue body. An LLM will automatically update the webpage based on your instructions.

The webpage `index.html` is hosted at https://mnaei.github.io/PromptPlace2/

## How It Works

1. **Issue Submission**: Users submit a GitHub issue with the label `prompt` which includes natural language instructions in the issue body.

2. **Triggering the Workflow**: A GitHub Actions workflow (`.github/workflows/llm-editor.yml`) is triggered by the labeled issue. This workflow:
   - Checks out the repository.
   - Sets up the Python environment.
   - Runs the `llm_editor.py` script.

3. **Processing the Request**: The `llm_editor.py` script performs the following:
   - **Fetch Current HTML**: Retrieves the current `index.html` file from the repository.
   - **Construct Prompt**: Combines the current HTML with the natural language instructions provided in the issue.
   - **Call Gemini API**: Sends the prompt to the Gemini API to generate updated HTML/CSS/JS code.
   - **Extract Valid HTML**: Uses a Python library to extract only valid HTML tags from the API response.
   - **Commit Changes**: Commits the updated `index.html` directly to the `main` branch.
   - **Comment on Issue**: Posts a comment on the issue containing a link to the commit.

4. **Concurrency Control**: The GitHub Actions workflow leverages the concurrency features to prevent merge conflict.

## LLM

Uses Gemini 2.0 Flash which is rate limited to 15 Request Per Minute. 


