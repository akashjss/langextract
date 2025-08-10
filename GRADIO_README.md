# LangExtract Gradio Interface

A user-friendly web interface for LangExtract built with Gradio. Extract structured information from text using Large Language Models through an intuitive web interface.

![LangExtract Gradio Interface](https://img.shields.io/badge/interface-gradio-orange)

## 🚀 Quick Start

### Installation

```bash
# Install LangExtract with Gradio support
pip install -e ".[gradio]"

# Or install Gradio separately
pip install gradio>=4.0.0
```

### Launch the Interface

```bash
# Simple launcher
python launch_gradio.py

# Or run directly
python gradio_app.py
```

The interface will be available at `http://localhost:7860`

## 🎯 Features

### ✨ Core Functionality
- **Interactive Text Extraction**: Upload or paste text for analysis
- **Flexible Example System**: Define extraction patterns with examples
- **Real-time Visualization**: See extracted entities highlighted in context
- **Multi-model Support**: Works with Gemini, OpenAI, and Ollama models
- **Advanced Configuration**: Fine-tune extraction parameters

### 📊 Interface Tabs

1. **📝 Extraction**: Main interface for input text and examples
2. **⚙️ Model Settings**: Configure model parameters and API keys
3. **🚀 Results**: View extraction results and interactive visualizations
4. **ℹ️ Help & Examples**: Documentation and example templates

## 🛠️ Usage Guide

### 1. Input Your Text
Paste the text you want to extract information from in the "Text to Extract From" field.

### 2. Define Extraction Task
Write clear instructions about what you want to extract:
```
Extract characters, emotions, and relationships in order of appearance.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.
```

### 3. Provide Examples
Examples are **required**. Use the built-in templates or create your own:

```
---
Text: ROMEO. But soft! What light through yonder window breaks?

Extractions:
- class: character, text: ROMEO, attributes: {"emotional_state": "wonder"}
- class: emotion, text: But soft!, attributes: {"feeling": "gentle awe"}
---
```

### 4. Configure Model
- **API Key**: Required for cloud models (Gemini, OpenAI)
- **Model Selection**: Choose from available models
- **Parameters**: Adjust temperature, chunk size, etc.

### 5. Extract & Visualize
Click "🔍 Extract Information" to run the extraction and view:
- **Summary**: Overview of results
- **JSON Data**: Structured extraction data
- **Visualization**: Interactive highlighting

## 🎨 Example Templates

The interface includes three built-in templates:

### 📚 Romeo & Juliet (Literary Analysis)
Extracts characters, emotions, and relationships from literary text.

### 🏥 Medical (Healthcare)
Extracts medications, conditions, and instructions from clinical text.

### 💼 Business (Corporate)
Extracts people, companies, funding, and investors from business text.

## 🌐 Ollama Support (Local Models)

The Gradio interface now includes comprehensive support for local Ollama models, enabling **completely private** text extraction without API keys!

### ✨ Key Features

- **🔍 Ollama Connection Test**: Built-in connectivity testing with model detection
- **🎯 Auto-Configuration**: Automatically detects local models and sets optimal parameters
- **🚫 No API Key Required**: Complete privacy with local processing
- **📋 20+ Supported Models**: From tiny 1B to large 70B parameter models
- **⚙️ Smart Settings**: Automatic parameter optimization for Ollama models

### 🚀 Quick Ollama Setup

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

3. **Install a Model**:
   ```bash
   # Small and fast (good for testing)
   ollama pull gemma2:2b        # 1.6 GB
   
   # Balanced performance
   ollama pull llama3.2:3b      # 2.0 GB
   ollama pull phi3:mini        # 2.3 GB
   
   # High quality
   ollama pull llama3.1:8b      # 4.7 GB
   ollama pull mistral:7b       # 4.1 GB
   ```

4. **Use in Gradio**:
   - Select any local model (e.g., "gemma2:2b") from the dropdown
   - Leave API key field empty
   - Click "🔍 Test Ollama" to verify connection
   - Extract with complete privacy!

### 📊 Supported Ollama Models

| Model Family | Sizes Available | Best For | Memory Usage |
|--------------|-----------------|----------|--------------|
| **Gemma2** | 2B, 9B, 27B | General use, fast | 2-16 GB |
| **Llama 3.x** | 1B, 3B, 8B, 70B | High quality | 1-40 GB |
| **Mistral** | 7B, Nemo 12B | Multilingual | 5-8 GB |
| **Phi3** | Mini (3.8B) | Efficient | 3-4 GB |
| **Qwen2.5** | 7B, 14B | Code + Text | 5-10 GB |
| **CodeLlama** | 7B, 13B, 34B | Code generation | 5-20 GB |
| **DeepSeek** | R1 8B | Reasoning | 6-8 GB |
| **TinyLlama** | 1.1B | Minimal resources | 1 GB |

### 🔧 Ollama Configuration

The Gradio interface automatically handles Ollama configuration:

- **Default URL**: `http://localhost:11434`
- **Connection Test**: Built-in "🔍 Test Ollama" button
- **Auto-Detection**: Automatically identifies local vs cloud models
- **Optimized Settings**: 
  - `fence_output=False` (Ollama works better without code fences)
  - `use_schema_constraints=False` (not supported by Ollama)
- **Model Discovery**: Shows available models when testing connection

### 💡 Ollama Tips

1. **Start with Small Models**: Try `gemma2:2b` or `llama3.2:1b` first
2. **Check Memory**: Ensure you have enough RAM for larger models
3. **Test Connection**: Always use "🔍 Test Ollama" before extraction
4. **Model Selection**: Choose based on your needs:
   - **Speed**: tinyllama:1.1b, gemma2:2b
   - **Balance**: llama3.2:3b, phi3:mini
   - **Quality**: llama3.1:8b, mistral:7b
   - **Code**: codellama:7b, qwen2.5:7b

## ⚙️ Model Configuration

### Recommended Settings

| Model | Use Case | Settings |
|-------|----------|----------|
| `gemini-2.5-flash` | General use (fast + accurate) | Default settings |
| `gemini-2.5-pro` | Complex reasoning | Higher temperature |
| `gpt-4o` | OpenAI ecosystem | `fence_output=True`, `use_schema_constraints=False` |
| `gemma2:2b` | Local/private | Requires Ollama |

### Parameter Guide

- **Temperature**: 0.0 (deterministic) to 1.0 (creative)
- **Max Characters per Chunk**: Smaller for accuracy, larger for speed
- **Extraction Passes**: Multiple passes improve recall but increase cost
- **Max Workers**: Parallel processing (more = faster)

## 🔐 API Keys

### Getting API Keys

1. **Gemini Models**: [AI Studio](https://aistudio.google.com/app/apikey)
2. **OpenAI Models**: [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Local Models**: No API key needed (use Ollama)

### Setting API Keys

**Option 1: Environment Variable**
```bash
export LANGEXTRACT_API_KEY="your-api-key"
```

**Option 2: .env File**
```bash
echo "LANGEXTRACT_API_KEY=your-api-key" >> .env
```

**Option 3: Web Interface**
Enter your API key in the "Model Settings" tab (not saved permanently).

## 🌐 Local LLM Support

Use local models with Ollama for privacy:

```bash
# Install and start Ollama
ollama pull gemma2:2b
ollama serve

# In the interface, select "gemma2:2b" as model
# No API key required
```

## 📋 Example Workflows

### Workflow 1: Medical Text Analysis
1. Select "🏥 Medical" template
2. Paste clinical note or medical report
3. Adjust extraction passes for better recall
4. Extract and review results

### Workflow 2: Document Intelligence
1. Create custom examples for your domain
2. Use multiple extraction passes
3. Fine-tune chunk size for your document length
4. Export results as JSON

### Workflow 3: Research & Analysis
1. Use literary or academic text
2. Create domain-specific entity types
3. Include rich attributes for context
4. Visualize patterns in the results

## 🐛 Troubleshooting

### Common Issues

**"Error: Please provide an API key"**
- Set `LANGEXTRACT_API_KEY` environment variable
- Or enter API key in the interface

**"Could not parse examples"**
- Check example format with `---` separators
- Ensure proper indentation
- Use built-in templates as reference

**"No extractions found"**
- Review your examples - they guide the model
- Try different model or adjust temperature
- Check if prompt description is clear

**Gradio import error**
```bash
pip install gradio>=4.0.0
```

### Performance Tips

- Use `gemini-2.5-flash` for best speed/quality balance
- Smaller chunks (500-1000 chars) for better accuracy
- Multiple workers for faster processing
- Single extraction pass unless you need higher recall

## 🔗 Integration

The Gradio interface can be embedded in other applications:

```python
from gradio_app import create_gradio_interface

# Create interface
interface = create_gradio_interface()

# Launch with custom settings
interface.launch(
    server_name="0.0.0.0",
    server_port=8080,
    share=True  # Creates public link
)
```

## 📄 License

This Gradio interface is part of LangExtract and follows the same Apache 2.0 license.

---

**Happy Extracting! 🎉**

For more information about LangExtract, see the [main README](README.md).
