#!/usr/bin/env python3
"""
Gradio Web Interface for LangExtract

This module provides a user-friendly web interface for LangExtract using Gradio.
Users can extract structured information from text using LLMs through an intuitive
web interface.
"""

import json
import os
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

import gradio as gr
import requests
import langextract as lx

# PDF processing imports
try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


def create_extraction_example(
    text: str,
    extraction_class: str,
    extraction_text: str,
    attributes: str = ""
) -> lx.data.Extraction:
    """Create an Extraction object from user inputs."""
    attrs = {}
    if attributes.strip():
        try:
            attrs = json.loads(attributes)
        except json.JSONDecodeError:
            # Try to parse as key:value pairs
            attrs = {}
            for line in attributes.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    attrs[key.strip()] = value.strip()

    return lx.data.Extraction(
        extraction_class=extraction_class,
        extraction_text=extraction_text,
        attributes=attrs if attrs else None
    )


def parse_examples_from_text(examples_text: str) -> List[lx.data.ExampleData]:
    """Parse examples from formatted text input."""
    examples = []

    # Split by sections starting with "Example" or "Text:"
    sections = examples_text.split('---')

    for section in sections:
        if not section.strip():
            continue

        lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
        if not lines:
            continue

        # Find the text
        text = ""
        extractions = []

        current_mode = None
        current_extraction = {}

        for line in lines:
            if line.lower().startswith('text:'):
                text = line[5:].strip()
                current_mode = 'text'
            elif line.lower().startswith('extractions:'):
                current_mode = 'extractions'
            elif current_mode == 'extractions' and line.startswith('- '):
                # Parse extraction line like "- class: character, text: ROMEO, attributes: {...}"
                extraction_data = line[2:].strip()

                # Simple parsing
                parts = extraction_data.split(', ')
                extraction_info = {}

                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        if key == 'attributes' and value.startswith('{'):
                            try:
                                extraction_info[key] = json.loads(value)
                            except json.JSONDecodeError:
                                extraction_info[key] = value
                        else:
                            extraction_info[key] = value

                if 'class' in extraction_info and 'text' in extraction_info:
                    attrs = extraction_info.get('attributes', {})
                    extractions.append(lx.data.Extraction(
                        extraction_class=extraction_info['class'],
                        extraction_text=extraction_info['text'],
                        attributes=attrs if attrs else None
                    ))

        if text and extractions:
            examples.append(lx.data.ExampleData(text=text, extractions=extractions))

    return examples


def run_extraction(
    input_text: str,
    prompt_description: str,
    examples_text: str,
    model_id: str,
    api_key: str,
    model_url: str,
    temperature: float,
    max_char_buffer: int,
    extraction_passes: int,
    max_workers: int,
    use_schema_constraints: bool,
    fence_output: bool
) -> Tuple[str, str, str]:
    """Run the extraction and return results."""

    if not input_text or not input_text.strip():
        return "Error: Please provide input text", "", ""

    if not prompt_description or not prompt_description.strip():
        return "Error: Please provide a prompt description", "", ""

    if not examples_text or not examples_text.strip():
        return "Error: Please provide at least one example", "", ""

    try:
        # Parse examples
        examples = parse_examples_from_text(examples_text)

        if not examples:
            return "Error: Could not parse examples. Please check the format.", "", ""

        # Detect if this is an Ollama model (local model)
        ollama_patterns = [
            'gemma', 'llama', 'mistral', 'mixtral', 'phi', 'qwen',
            'deepseek', 'command-r', 'starcoder', 'codellama',
            'codegemma', 'tinyllama', 'wizardcoder'
        ]
        is_ollama_model = any(pattern in model_id.lower() for pattern in ollama_patterns)

        # Set API key for cloud models
        if not is_ollama_model:
            if api_key and api_key.strip():
                os.environ['LANGEXTRACT_API_KEY'] = api_key.strip()
            elif 'LANGEXTRACT_API_KEY' not in os.environ:
                return "Error: Please provide an API key for cloud models (Gemini/OpenAI) or use a local Ollama model", "", ""

        # Set model URL for Ollama models
        if is_ollama_model and (not model_url or not model_url.strip()):
            model_url = "http://localhost:11434"

        # Configure parameters based on model type
        extract_kwargs = {
            "text_or_documents": input_text,
            "prompt_description": prompt_description,
            "examples": examples,
            "model_id": model_id,
            "temperature": temperature,
            "max_char_buffer": max_char_buffer,
            "extraction_passes": extraction_passes,
            "max_workers": max_workers,
            "debug": True
        }

        if is_ollama_model:
            # Ollama-specific settings
            extract_kwargs.update({
                "model_url": model_url.strip() if model_url else "http://localhost:11434",
                "fence_output": False,  # Ollama works better without fencing
                "use_schema_constraints": False  # Ollama doesn't support schema constraints
            })
        else:
            # Cloud model settings
            extract_kwargs.update({
                "use_schema_constraints": use_schema_constraints,
                "fence_output": fence_output
            })

            # OpenAI models need specific settings
            if model_id.startswith('gpt-'):
                extract_kwargs.update({
                    "fence_output": True,
                    "use_schema_constraints": False
                })

        # Run extraction
        result = lx.extract(**extract_kwargs)

        # Format results
        if result.extractions:
            extractions_json = []
            for extraction in result.extractions:
                extraction_dict = {
                    "extraction_class": extraction.extraction_class,
                    "extraction_text": extraction.extraction_text,
                    "attributes": extraction.attributes,
                }
                if extraction.char_interval:
                    extraction_dict["char_interval"] = {
                        "start_pos": extraction.char_interval.start_pos,
                        "end_pos": extraction.char_interval.end_pos
                    }
                extractions_json.append(extraction_dict)

            results_text = json.dumps(extractions_json, indent=2)

            # Create visualization
            try:
                viz_html = lx.visualize(result)
                if isinstance(viz_html, str):
                    visualization = viz_html
                else:
                    # If it's an IPython HTML object, get the data
                    visualization = str(viz_html.data) if hasattr(viz_html, 'data') else str(viz_html)
            except Exception as viz_error:
                visualization = f"<p>Visualization error: {str(viz_error)}</p>"

            model_type = "🌐 Local (Ollama)" if is_ollama_model else "☁️ Cloud"
            summary = f"✅ Extraction completed successfully!\n"
            summary += f"Found {len(result.extractions)} extractions\n"
            summary += f"Text length: {len(input_text)} characters\n"
            summary += f"Model: {model_id} ({model_type})\n"
            if is_ollama_model:
                summary += f"Ollama server: {model_url.strip()}"
            else:
                summary += f"API provider: {'OpenAI' if model_id.startswith('gpt-') else 'Google Gemini'}"

            return summary, results_text, visualization
        else:
            return "No extractions found", "[]", "<p>No extractions to visualize</p>"

    except Exception as e:
        error_msg = f"Error during extraction: {str(e)}\n\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg, "", ""


def get_example_romeo_juliet():
    """Get Romeo and Juliet example."""
    return """---
Text: ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.

Extractions:
- class: character, text: ROMEO, attributes: {"emotional_state": "wonder"}
- class: emotion, text: But soft!, attributes: {"feeling": "gentle awe"}
- class: relationship, text: Juliet is the sun, attributes: {"type": "metaphor"}
---"""


def get_example_medical():
    """Get medical example."""
    return """---
Text: Patient was prescribed Lisinopril 10mg once daily for hypertension. Take with food.

Extractions:
- class: medication, text: Lisinopril, attributes: {"dosage": "10mg", "frequency": "once daily"}
- class: condition, text: hypertension, attributes: {"type": "cardiovascular"}
- class: instruction, text: Take with food, attributes: {"type": "administration"}
---"""


def get_example_business():
    """Get business example."""
    return """---
Text: John Smith, CEO of TechCorp, announced a $50M funding round led by Venture Capital Partners.

Extractions:
- class: person, text: John Smith, attributes: {"role": "CEO", "company": "TechCorp"}
- class: company, text: TechCorp, attributes: {"type": "technology"}
- class: funding, text: $50M funding round, attributes: {"amount": "50000000", "type": "funding_round"}
- class: investor, text: Venture Capital Partners, attributes: {"role": "lead_investor"}
---"""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    if not PDF_SUPPORT:
        raise ImportError("pypdf library not installed. Install with: pip install pypdf")

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            return "Error: Could not extract text from PDF. The PDF might be image-based or protected."

        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a Word document."""
    if not DOCX_SUPPORT:
        raise ImportError("python-docx library not installed. Install with: pip install python-docx")

    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        if not text.strip():
            return "Error: Could not extract text from Word document."

        return text.strip()
    except Exception as e:
        return f"Error reading Word document: {str(e)}"


def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats."""
    if not file_path:
        return "Error: No file provided"

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            if file_ext == '.doc':
                return "Error: .doc files not supported. Please convert to .docx format."
            return extract_text_from_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return f"Error: Unsupported file format '{file_ext}'. Supported: .pdf, .docx, .txt, .md"
    except Exception as e:
        return f"Error processing file: {str(e)}"


def process_uploaded_file(file) -> tuple[str, str]:
    """Process uploaded file and extract text."""
    if file is None:
        return "", "No file uploaded"

    try:
        # Get file information
        file_path = file.name
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # Check file size (limit to 10MB)
        if file_size > 10 * 1024 * 1024:
            return "", f"Error: File '{file_name}' is too large. Maximum size is 10MB."

        # Extract text
        extracted_text = extract_text_from_file(file_path)

        if extracted_text.startswith("Error:"):
            return "", extracted_text

        # Success message
        status_msg = f"✅ Successfully extracted text from '{file_name}'\n"
        status_msg += f"📄 File size: {file_size:,} bytes\n"
        status_msg += f"📝 Text length: {len(extracted_text):,} characters\n"
        status_msg += f"📊 Word count: {len(extracted_text.split()):,} words"

        return extracted_text, status_msg

    except Exception as e:
        return "", f"Error processing file: {str(e)}"


def test_ollama_connection(model_url: str) -> str:
    """Test connection to Ollama server and return status."""
    if not model_url or not model_url.strip():
        model_url = "http://localhost:11434"

    try:
        # Test basic connection
        response = requests.get(f"{model_url.strip()}/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()

            # Try to list available models
            models_response = requests.get(f"{model_url.strip()}/api/tags", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = [model['name'] for model in models_data.get('models', [])]

                if models:
                    available_models = ', '.join(models[:5])  # Show first 5 models
                    if len(models) > 5:
                        available_models += f" (and {len(models)-5} more)"

                    return f"✅ Ollama server connected!\n" \
                           f"Version: {version_info.get('version', 'unknown')}\n" \
                           f"Available models: {available_models}\n" \
                           f"Server URL: {model_url.strip()}"
                else:
                    return f"⚠️ Ollama server connected but no models installed.\n" \
                           f"Install a model with: ollama pull gemma2:2b\n" \
                           f"Server URL: {model_url.strip()}"
            else:
                return f"✅ Ollama server connected but couldn't list models.\n" \
                       f"Version: {version_info.get('version', 'unknown')}\n" \
                       f"Server URL: {model_url.strip()}"
        else:
            return f"❌ Ollama server responded with error: {response.status_code}\n" \
                   f"Server URL: {model_url.strip()}"

    except requests.exceptions.ConnectionError:
        return f"❌ Cannot connect to Ollama server.\n" \
               f"Make sure Ollama is running: ollama serve\n" \
               f"Server URL: {model_url.strip()}"
    except requests.exceptions.Timeout:
        return f"❌ Ollama server connection timeout.\n" \
               f"Server might be slow or unresponsive.\n" \
               f"Server URL: {model_url.strip()}"
    except Exception as e:
        return f"❌ Error testing Ollama connection: {str(e)}\n" \
               f"Server URL: {model_url.strip()}"


def create_gradio_interface():
    """Create the Gradio interface."""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .gr-button {
        background: linear-gradient(45deg, #4285f4, #34a853) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button:hover {
        background: linear-gradient(45deg, #3367d6, #2d8f47) !important;
    }
    .extraction-results {
        font-family: monospace;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    """

    with gr.Blocks(
        title="LangExtract - Structured Information Extraction",
        css=custom_css,
        theme=gr.themes.Ocean()
    ) as interface:

        gr.Markdown("""
        # 🔍 LangExtract - Interactive Extraction Interface

        Extract structured information from text using Large Language Models.
        Define your extraction task with examples and let LangExtract find the information you need!

        **Need an API Key?** Get one from [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models.
        """)

        with gr.Tab("📝 Extraction"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Input Text")

                    # File upload section
                    with gr.Row():
                        file_upload = gr.File(
                            label="📁 Upload Document (PDF, DOCX, TXT)",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            type="filepath"
                        )

                    upload_status = gr.Textbox(
                        label="Upload Status",
                        lines=3,
                        interactive=False,
                        visible=False
                    )

                    input_text = gr.Textbox(
                        label="Text to Extract From",
                        placeholder="Enter text manually or upload a document above...",
                        lines=8,
                        max_lines=20
                    )

                    gr.Markdown("### Extraction Instructions")
                    prompt_description = gr.Textbox(
                        label="Prompt Description",
                        placeholder="Describe what you want to extract (e.g., 'Extract characters, emotions, and relationships...')",
                        lines=3,
                        value="Extract entities in order of appearance. Use exact text for extractions. Do not paraphrase or overlap entities. Provide meaningful attributes for each entity to add context."
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Examples (Required)")
                    gr.Markdown("Provide examples in the format shown. Use `---` to separate multiple examples.")

                    with gr.Row():
                        example_romeo_btn = gr.Button("📚 Romeo & Juliet", size="sm")
                        example_medical_btn = gr.Button("🏥 Medical", size="sm")
                        example_business_btn = gr.Button("💼 Business", size="sm")

                    examples_text = gr.Textbox(
                        label="Examples",
                        placeholder="""---
Text: Your example text here

Extractions:
- class: entity_type, text: exact text from above, attributes: {"key": "value"}
- class: another_type, text: another exact text, attributes: {"key": "value"}
---""",
                        lines=12,
                        max_lines=30
                    )

        with gr.Tab("⚙️ Model Settings"):
            with gr.Row():
                with gr.Column():
                    model_id = gr.Dropdown(
                        label="Model",
                        choices=[
                            # Cloud models (require API key)
                            "gemini-2.5-flash",
                            "gemini-2.5-pro",
                            "gemini-1.5-flash",
                            "gemini-1.5-pro",
                            "gpt-4o",
                            "gpt-4o-mini",
                            # Local Ollama models (no API key needed)
                            "gemma2:2b",
                            "gemma2:9b",
                            "gemma2:27b",
                            "llama3.2:1b",
                            "llama3.2:3b",
                            "llama3.1:8b",
                            "llama3.1:70b",
                            "mistral:7b",
                            "mistral-nemo:12b",
                            "mixtral:8x7b",
                            "phi3:mini",
                            "qwen2.5:7b",
                            "qwen2.5:14b",
                            "deepseek-r1:8b",
                            "codellama:7b",
                            "codegemma:2b",
                            "tinyllama:1.1b"
                        ],
                        value="gemini-2.5-flash",
                        info="Choose your model. Gemini for cloud, local models (gemma2, llama, etc.) for privacy."
                    )

                    api_key = gr.Textbox(
                        label="API Key (Required for cloud models)",
                        placeholder="Enter your API key here (not needed for local Ollama models)",
                        type="password",
                        info="Get API keys from AI Studio (Gemini) or OpenAI Platform. Leave empty for Ollama."
                    )

                    with gr.Row():
                        model_url = gr.Textbox(
                            label="Ollama Server URL (for local models)",
                            placeholder="http://localhost:11434",
                            value="http://localhost:11434",
                            info="URL of your Ollama server. Only used for local models like gemma2, llama, etc.",
                            scale=3
                        )
                        test_ollama_btn = gr.Button("🔍 Test Ollama", scale=1, size="sm")

                    ollama_status = gr.Textbox(
                        label="Ollama Status",
                        lines=4,
                        interactive=False,
                        visible=False
                    )

                with gr.Column():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        info="Higher values increase randomness"
                    )

                    max_char_buffer = gr.Slider(
                        label="Max Characters per Chunk",
                        minimum=500,
                        maximum=5000,
                        value=1000,
                        step=100,
                        info="Smaller chunks for better accuracy, larger for efficiency"
                    )

                with gr.Column():
                    extraction_passes = gr.Slider(
                        label="Extraction Passes",
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        info="Multiple passes improve recall but increase cost"
                    )

                    max_workers = gr.Slider(
                        label="Max Workers",
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        info="Parallel processing workers"
                    )

                with gr.Column():
                    use_schema_constraints = gr.Checkbox(
                        label="Use Schema Constraints",
                        value=True,
                        info="Enable structured outputs (recommended for Gemini)"
                    )

                    fence_output = gr.Checkbox(
                        label="Fence Output",
                        value=False,
                        info="Expect ```json or ```yaml fenced output"
                    )

        with gr.Tab("🚀 Results"):
            gr.Markdown("### Run Extraction")

            extract_btn = gr.Button(
                "🔍 Extract Information",
                variant="primary",
                size="lg"
            )

            gr.Markdown("### Extraction Summary")
            summary_output = gr.Textbox(
                label="Summary",
                lines=4,
                interactive=False
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Extracted Data (JSON)")
                    results_output = gr.Code(
                        label="Results",
                        language="json",
                        lines=15
                    )

                with gr.Column():
                    gr.Markdown("### Interactive Visualization")
                    visualization_output = gr.HTML(
                        label="Visualization"
                    )

        with gr.Tab("ℹ️ Help & Examples"):
            gr.Markdown("""
            ## How to Use LangExtract

                        ### 1. Input Text
            You can provide text in two ways:

            **📁 Upload Documents:**
            - **PDF files** (.pdf) - Automatic text extraction
            - **Word documents** (.docx) - Full text extraction
            - **Text files** (.txt, .md) - Direct import
            - **Size limit**: 10MB maximum

            **✏️ Manual Entry:**
            - Paste or type text directly
            - Clinical notes or medical reports
            - Literary texts, business documents
            - News articles, any unstructured text

            ### 2. Define Your Task
            Write clear instructions about what you want to extract:
            - Be specific about entity types
            - Mention if you want attributes
            - Specify extraction order if important

            ### 3. Provide Examples
            Examples are **required** and guide the model. Format:
            ```
            ---
            Text: Your example text here

            Extractions:
            - class: entity_type, text: exact text span, attributes: {"key": "value"}
            - class: another_type, text: another span, attributes: {"key2": "value2"}
            ---
            ```

            ### 4. Choose Model & Settings
            - **Gemini models**: Best for structured extraction with schema constraints
            - **GPT models**: Require `fence_output=True` and `use_schema_constraints=False`
            - **Local models**: Use Ollama models for privacy

            ### 5. View Results
            - **Summary**: Overview of extraction results
            - **JSON Data**: Structured extraction data
            - **Visualization**: Interactive highlighting of extractions

            ## Tips for Better Results
            - Use high-quality, representative examples
            - Be consistent in your extraction class names
            - Include meaningful attributes that add context
            - For long documents, consider multiple extraction passes
            - Smaller chunks (max_char_buffer) often give better accuracy

            ## Model Comparison

            ### ☁️ Cloud Models (Require API Key)
            | Model | Best For | Provider | Notes |
            |-------|----------|----------|-------|
            | gemini-2.5-flash | Speed + Quality | Google AI | Recommended default |
            | gemini-2.5-pro | Complex reasoning | Google AI | Best for difficult tasks |
            | gpt-4o | OpenAI ecosystem | OpenAI | Requires fence_output=True |
            | gpt-4o-mini | Cost efficiency | OpenAI | Faster, cheaper GPT |

            ### 🌐 Local Models (No API Key - Ollama)
            | Model | Size | Best For | Memory |
            |-------|------|----------|--------|
            | gemma2:2b | 2B | Fast inference | 2-4 GB |
            | gemma2:9b | 9B | Balanced performance | 6-8 GB |
            | llama3.2:1b | 1B | Ultra-fast | 1-2 GB |
            | llama3.2:3b | 3B | Good quality | 2-3 GB |
            | llama3.1:8b | 8B | High quality | 6-8 GB |
            | mistral:7b | 7B | Multilingual | 5-7 GB |
            | phi3:mini | 3.8B | Efficient | 3-4 GB |
            | qwen2.5:7b | 7B | Code + Text | 5-7 GB |
            | codellama:7b | 7B | Code generation | 5-7 GB |
            | tinyllama:1.1b | 1B | Minimal resources | 1 GB |

            ## 🚀 Ollama Setup Guide

            ### Quick Setup
            ```bash
            # 1. Install Ollama
            curl -fsSL https://ollama.com/install.sh | sh

            # 2. Start Ollama server
            ollama serve

            # 3. Pull a model (in another terminal)
            ollama pull gemma2:2b

            # 4. Test in LangExtract Gradio
            # Select "gemma2:2b" model and run extraction!
            ```

            ### Popular Model Downloads
            ```bash
            # Small models (good for testing)
            ollama pull tinyllama:1.1b      # 637 MB
            ollama pull gemma2:2b           # 1.6 GB
            ollama pull llama3.2:1b         # 1.3 GB

            # Medium models (balanced)
            ollama pull llama3.2:3b         # 2.0 GB
            ollama pull phi3:mini           # 2.3 GB
            ollama pull gemma2:9b           # 5.5 GB

            # Large models (best quality)
            ollama pull llama3.1:8b         # 4.7 GB
            ollama pull mistral:7b          # 4.1 GB
            ollama pull qwen2.5:7b          # 4.5 GB
            ```

            ### Ollama Configuration
            - **Default URL**: `http://localhost:11434`
            - **No API Key Required**: Just select an Ollama model
            - **Privacy**: All processing happens locally
            - **Settings**: fence_output and schema_constraints auto-disabled
            """)

                # Event handlers
        example_romeo_btn.click(
            fn=get_example_romeo_juliet,
            outputs=examples_text
        )

        example_medical_btn.click(
            fn=get_example_medical,
            outputs=examples_text
        )

        example_business_btn.click(
            fn=get_example_business,
            outputs=examples_text
        )

        def test_and_show_ollama(url):
            """Test Ollama and return status with visibility."""
            status = test_ollama_connection(url)
            return status, gr.update(visible=True)

        def handle_file_upload(file):
            """Handle file upload and extract text."""
            if file is None:
                return "", gr.update(visible=False)

            extracted_text, status_msg = process_uploaded_file(file)

            if status_msg.startswith("Error:"):
                return "", gr.update(value=status_msg, visible=True)
            else:
                return extracted_text, gr.update(value=status_msg, visible=True)

        # Event handlers
        file_upload.change(
            fn=handle_file_upload,
            inputs=file_upload,
            outputs=[input_text, upload_status]
        )

        test_ollama_btn.click(
            fn=test_and_show_ollama,
            inputs=model_url,
            outputs=[ollama_status, ollama_status]
        )

        extract_btn.click(
            fn=run_extraction,
            inputs=[
                input_text,
                prompt_description,
                examples_text,
                model_id,
                api_key,
                model_url,
                temperature,
                max_char_buffer,
                extraction_passes,
                max_workers,
                use_schema_constraints,
                fence_output
            ],
            outputs=[summary_output, results_output, visualization_output]
        )

    return interface


def main():
    """Main function to launch the Gradio app."""
    print("🚀 Starting LangExtract Gradio Interface...")

    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
        debug=True,
        inbrowser=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
